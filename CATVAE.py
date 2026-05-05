import torch
import random
import numpy as np
from itertools import product

from torch import nn
from info_nce import InfoNCE
from sklearn.metrics import normalized_mutual_info_score
from utils import (augment_data, compute_purity,
                   evaluate_state_based_classifier_with_windows, compute_state_label_metrics, summarize_seed_results)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CategoricalVAE(nn.Module):
    """
    Categorical variational autoencoder with a discrete latent state space.
    """
    def __init__(self, hparams) -> None:
        super(CategoricalVAE, self).__init__()

        # Parameters from hparams dictionary
        self.in_dim = hparams["IN_DIM"]
        self.enc_out_dim = hparams["ENC_OUT_DIM"]
        self.dec_out_dim = hparams["DEC_OUT_DIM"]
        self.categorical_dim = hparams["CATEGORICAL_DIM"]
        self.temp = hparams["TEMPERATURE"]
        self.beta = hparams["BETA"]

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim * 2),
            nn.ReLU(),
            nn.Linear(self.in_dim * 2, self.in_dim * 2),
            nn.ReLU(),
            nn.Linear(self.in_dim * 2, self.enc_out_dim))
        self.fc_z_cat = nn.Linear(self.enc_out_dim, self.categorical_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.categorical_dim, self.in_dim * 2),
            nn.ReLU(),
            nn.Linear(self.in_dim * 2, self.in_dim * 2),
            nn.ReLU(),
            nn.Linear(self.in_dim * 2, self.dec_out_dim))
        self.fc_mu_x = nn.Linear(self.dec_out_dim, self.in_dim)
        self.fc_logvar_x = nn.Linear(self.dec_out_dim, self.in_dim)

        # Categorical prior
        self.pz = torch.distributions.OneHotCategorical(
            1. / self.categorical_dim * torch.ones(1, self.categorical_dim, device=device))

        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.)[0], requires_grad=True)


    def forward(self, x: torch.Tensor):
        """
        Compute the CatVAE loss and return the posterior logits.

        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            loss: Scalar loss tensor.
            logits: Categorical posterior logits of shape [batch_size, categorical_dim].

        """
        logits, posterior, reconstruction_dist, _ = self.computation(x)
        loss_dct = self.loss_function(
            x=x,
            posterior=posterior,
            reconstruction_dist=reconstruction_dist,
        )

        return loss_dct['Loss'], logits


    def encode(self, input_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.

        Args:
            input_data: The data of shape [batch_size, input_dim].

        Return:
            logits: Categorical posterior logits.
            encoder_output: Output of the encoder before the categorical projection.
        """
        encoder_output = self.encoder(input_data)
        logits = self.fc_z_cat(torch.flatten(encoder_output, start_dim=1))
        logits = logits.view(-1, self.categorical_dim)
        return logits, encoder_output


    def decode(self, z: torch.Tensor) -> tuple:
        """
        Decode latent categorical samples into a diagonal Gaussian distribution p(x|z).

        Args:
            z: Relaxed or hard one-hot latent state of shape [batch_size, categorical_dim].

        Returns:
            reconstruction_dist: Independent diagonal Gaussian distribution over x.
            mu: Mean of the decoder distribution.
            logvar: Log-variance of the decoder distribution.
        """
        result = self.decoder(z)

        mu = self.fc_mu_x(result)
        logvar = self.fc_logvar_x(result)

        # Clamp for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)

        reconstruction_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)

        return reconstruction_dist, mu, logvar


    def computation(self, x: torch.Tensor):
        """
        Shared forward computation used during training and evaluation.
        """
        # Compute parameters of the categorical distribution
        logits, _ = self.encode(x)

        # Create one hot categorical distribution object for use in loss function
        posterior = torch.distributions.OneHotCategorical(logits=logits)

        # Sample from the distribution
        z = self.sample_gumbel(logits=logits)

        # Reconstruct
        reconstruction_dist, _, _ = self.decode(z)

        return logits, posterior, reconstruction_dist, z


    def get_states(self, x: torch.Tensor):
        """
        Compute the discrete latent states by taking the argmax over posterior logits.

        Args:
            x: Input tensor of shape [batch:size, input_dim].

        Return:
            logits: Posterior logits.
            posterior: Categorical posterior q(z|x).
            reconstruction_dist: Decoder distribution p(x|z).
            z_relaxed: Relaxed Gumbel-Softmax sample.
            z_hard: Hard one-hot state assignment.
        """
        logits, _ = self.encode(x)

        posterior = torch.distributions.OneHotCategorical(logits=logits)
        z_relaxed = self.sample_gumbel(logits)

        state_indices = torch.argmax(logits, dim=1)
        z_hard = torch.zeros_like(logits)
        z_hard.scatter_(1, state_indices.unsqueeze(1), 1.0)

        reconstruction_dist, _, _ = self.decode(z_hard)

        return logits, posterior, reconstruction_dist, z_relaxed, z_hard


    def sample_gumbel(self, logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Sample a relaxed categorical variable using the Gumbel-Softmax trick.

        Args:
            logits: Categorical posterior logits.
            eps: Small constant for numerical stability.

        Returns:
            Relaxed one-hot sample.
        """
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        z = torch.nn.functional.softmax((logits + gumbel_noise) / self.temp, dim=-1)
        return z


    def loss_function(self, x: torch.Tensor,
                      posterior: torch.distributions.OneHotCategorical,
                      reconstruction_dist: torch.distributions.MultivariateNormal) -> dict:
        """"
        Compute the negative ELBO loss.

        Args:
            x: Input tensor.
            posterior: Categorical posterior q(z|x)-
            reconstruction_dist: Decoder distribution p(x|z).

        Returns:
            Dictionary containing total loss, reconstruction loss and KL loss.
        """
        # Compute reconstruction loss
        log_likelihood = reconstruction_dist.log_prob(x)
        reconstruction_loss = torch.mean(log_likelihood)

        # Compute the kl divergence for categorical dist
        prior_probs = torch.full(
            size=(1, self.categorical_dim),
            fill_value=1.0 / self.categorical_dim,
            device=x.device,
        )
        kl_categorical = torch.distributions.kl.kl_divergence(posterior,
                                                              torch.distributions.OneHotCategorical(probs=prior_probs))
        kl_categorical_batch = torch.mean(kl_categorical)

        loss = -reconstruction_loss + self.beta * kl_categorical_batch

        return {'Loss': loss, 'Reconstruction Loss': reconstruction_loss, 'KL Divergence': kl_categorical_batch}


    def discrete_comp(self, data: torch.Tensor):
        """
        Computes discrete state assignments and the set of unique assigned states.

        Args:
            data: Input tensor.

        Returns:
            categories: Integer state assignment for each sample.
            unique_states: List of unique one-hot state tuples.
        """
        # Get hard states
        _, _, _, _, z_states = self.get_states(data)
        categories = torch.argmax(z_states, dim=1)

        # Get unique states
        unique_states = {tuple(row.detach().cpu().tolist()) for row in z_states}

        return categories, list(unique_states)


def get_latent_data(model, valid_data):
    latent_data = []
    for data in valid_data:
       latent, _ = model.discrete_comp(data)
       latent_data.append(latent)
    return latent_data



def run_base_catvae(train_data: list, valid_data: list, train_states: list, valid_states:list , enc_out_dim: int,
                                  dec_out_dim: int, cat_dim: int, beta: float, temperature: float, seeds: list[int],
                                  input_dim: int, learning_rate: float = 0.005, num_epochs: int = 200) -> dict:
    """
    Run the CatVAE training for a fixed set of hyperparameters for several seeds.
    
    Args:
        train_data:     The training data.
        valid_data:     The validation data.
        train_states:   The training states.
        valid_states:   The validation states.
        enc_out_dim:    Dimension of the encoder.
        dec_out_dim:    Dimension of the decoder.
        cat_dim:        Dimension of the categorical distribution.
        beta:           KL-divergence weight.
        temperature:    Gumbel-Softmax temperature.
        seeds:          Random seeds.
        input_dim:      Input dimension of the data (number of channels).
        learning_rate:  Learning rate for training.
        num_epochs:     Number of epochs for training.

    Returns:
        Dictionary containing the hyperparameters, per-seed results and summary metrics.
    """
    # Initialize hparams dictionary
    hparams = {
        "IN_DIM": input_dim,
        "ENC_OUT_DIM": enc_out_dim,
        "DEC_OUT_DIM": dec_out_dim,
        "CATEGORICAL_DIM": cat_dim,
        "TEMPERATURE": temperature,
        "BETA": beta
    }

    seed_results = []

    # Train network for set of hyperparameters for different seeds
    for seed in seeds:
        print(f"Training base CatVAE with seed={seed}")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model
        model = CategoricalVAE(hparams)
        model.to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Get results
        loss, model = train_base_catvae(model, train_data, optimizer, num_epochs)
        metrics = evaluate_catvae_model(model, train_data, valid_data, train_states, valid_states)

        result = {"seed": seed, "loss": loss, **metrics}
        seed_results.append(result)

    summary = summarize_seed_results(seed_results)
    print(summary)

    return {
        "model": "catvae",
        "variant": "base",
        "hparams": hparams,
        "training": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "seeds": seeds,
        },
        "summary": summary,
    }


def train_base_catvae(model: torch.nn.Module, train_data: list, optimizer: torch.optim.Optimizer,
                     num_epochs: int) -> tuple[float, torch.nn.Module]:
    """
    Trains the base model of the CatVAE.

    Args:
        model:      The initialised model.
        train_data: List of training data.
        optimizer:  The optimiser.
        num_epochs: Number of epochs.

    Returns:
        model:      The trained model.
        loss:       The final loss.
    """
    # Set the model to training mode
    model.train()
    final_loss = float("nan")

    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        epoch_losses = []

        for data in train_data:
            # Clear previous gradients
            optimizer.zero_grad()

            # Move data to device
            data = data.to(device)

            # Forward pass
            loss, _ = model(data)

            # Backward pass
            loss.backward()

            # Take optimization step
            optimizer.step()

            epoch_losses.append(loss.item())

        final_loss = float(np.mean(epoch_losses))

    # Return the final loss and model
    return final_loss, model


def run_contrastive_catvae(train_data: list, valid_data: list, train_states: list, valid_states: list, noise: float,
                            info_temperature: float, seeds: list[int], input_dim: int, enc_out_dim: int = 4,
                            dec_out_dim: int = 32, cat_dim: int = 16, beta: float = 0.8, gumbel_temperature: float = 0.8,
                            learning_rate: float = 0.005, num_epochs: int = 100) -> dict:
    """
    Train and evaluate the contrastive CatVAE for a fixed hyperparameter configuration
    across multiple random seeds.

    Args:
        train_data:             Training data.
        valid_data:             Validation data.
        train_states:           Ground-truth states for the training data.
        valid_states:           Ground-truth states for the validation data.
        noise:                  Noise level for augmentation.
        info_temperature:       Temperature of the InfoNCE loss.
        seeds:                  Random seeds.
        input_dim:              Input dimensionality.
        enc_out_dim:            Encoder output dimension.
        dec_out_dim:            Decoder output dimension.
        cat_dim:                Number of categorical latent states.
        beta:                   KL-divergence weight.
        gumbel_temperature:     Gumbel-Softmax temperature.
        learning_rate:          Learning rate.
        num_epochs:             Number of training epochs.

    Returns:
        Dictionary containing the hyperparameters, per-seed results, and summary metrics.
    """
    hparams = {
        "IN_DIM": input_dim,
        "ENC_OUT_DIM": enc_out_dim,
        "DEC_OUT_DIM": dec_out_dim,
        "CATEGORICAL_DIM": cat_dim,
        "TEMPERATURE": gumbel_temperature,
        "BETA": beta,
    }

    seed_results = []

    for seed in seeds:
        print(f"Training contrastive CatVAE with seed={seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = CategoricalVAE(hparams).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss, model = train_contrastive_catvae(model=model, train_data=train_data, optimizer=optimizer, noise=noise,
                                              temp=info_temperature, num_epochs=num_epochs)

        metrics = evaluate_catvae_model(model=model, train_data=train_data, valid_data=valid_data,
                                        train_states=train_states, valid_states=valid_states)

        result = {"seed": seed, "loss": loss, **metrics}

        seed_results.append(result)

    summary = summarize_seed_results(seed_results)
    print(summary)

    return {
        "model": "catvae",
        "variant": "contrastive",
        "hparams": hparams,
        "training": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "seeds": seeds,
            "noise": noise,
            "info_temperature": info_temperature,
        },
        "summary": summary,
    }


def train_contrastive_catvae(model: nn.Module, train_data: list, optimizer: torch.optim.Optimizer, noise: float,
                            temp: float, num_epochs: int) -> tuple[float, torch.nn.Module]:
    """
    Trains the contrastive version of the given CatVAE model provided with the training data.
    The model is optimized using the CatVAE loss plus an InfoNCE loss between
    representations of original and augmented samples.

    Args:
        model:              The PyTorch model to be trained.
        train_data:         The list of time series datasets, where each dataset is represented by a tensor.
        optimizer:          The optimizer used during training.
        noise:              The noise of the augmentations.
        temp:               The temperature for the loss.
        num_epochs:         The number of training epochs.

    Returns:
        final_loss (float):         The final loss value from the last processed batch.
        model (torch.nn.Module):    The trained PyTorch model.
    """
    # Set the model to training mode
    model.train()

    # Set InfoNCE loss
    info_loss = InfoNCE(temperature=temp)
    final_loss = float("nan")

    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in train_data:
            # Clear previous gradients
            optimizer.zero_grad()

            # Move to device
            batch = batch.to(device)
            batch_aug = augment_data(batch, noise, device)

            # Forward pass
            # CatVAE loss
            catvae_loss, logits = model(batch)
            _, logits_aug = model(batch_aug)
 
            z = torch.nn.functional.normalize(logits, dim=1)
            z_aug = torch.nn.functional.normalize(logits_aug, dim=1)

            # Compute the contrastive loss
            contrastive_loss = info_loss(z, z_aug)
            loss = catvae_loss + contrastive_loss

            # Backward pass
            loss.backward()

            # Take optimization step
            optimizer.step()

            epoch_losses.append(loss.item())

        final_loss = float(np.mean(epoch_losses))

    # Return the final loss and model
    return final_loss, model


def evaluate_catvae_model(model: torch.nn.Module, train_data: list, valid_data: list, train_states: list,
                          valid_states: list, window_size: int = 128, step_size: int = 64) -> dict:
    """
    Evaluate a trained CatVAE model using clustering and downstream classification metrics.

    Args:
        model:          Trained CatVAE model.
        train_data:     Training data.
        valid_data:     Validation data.
        train_states:   Ground-truth training states.
        valid_states:   Ground-truth validation states.
        window_size:    Window size for downstream classifier.
        step_size:      Step size for downstream classifier.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Set model to evaluation mode
    model.eval()

    # Get states as lists
    train_latent_list = get_latent_data(model.to(device), train_data)
    valid_latent_list = get_latent_data(model.to(device), valid_data)

    # Compute purity
    purity_list = compute_purity(valid_latent_list, valid_states, device)
    purity = float(np.mean(purity_list))

    # Compute NMI
    nmis = []
    for latent, state in zip(valid_latent_list, valid_states):
        nmi = normalized_mutual_info_score(
            state.detach().cpu().numpy(),
            latent.detach().cpu().numpy(),
        )
        nmis.append(nmi)
    nmi = float(np.mean(nmis))

    # Compute entropy
    state_metrics, per_state_entropy = compute_state_label_metrics(
        valid_latent_list,
        valid_states,
    )

    # Compute classification results
    cls_acc, cls_macro_f1, _ = evaluate_state_based_classifier_with_windows(
        train_latent_list,
        train_states,
        valid_latent_list,
        valid_states,
        window_size=window_size,
        step_size=step_size,
        num_states=None,
        label_strategy="majority",
    )

    return {
        "purity": purity,
        "nmi": nmi,
        "avg_label_entropy": float(state_metrics["avg_label_entropy"]),
        "mutual_information": float(state_metrics["mutual_information"]),
        "entropy_labels": float(state_metrics["entropy_labels"]),
        "entropy_states": float(state_metrics["entropy_states"]),
        "per_state_entropy": per_state_entropy,
        "classification_accuracy": float(cls_acc),
        "classification_macro_f1": float(cls_macro_f1),
    }


def tune_base_catvae(train_data: list, valid_data: list, train_states: list, valid_states: list, seeds: list[int],
                     input_dim: int, enc_out_dims: list[int], dec_out_dims: list[int], cat_dims: list[int],
                     betas: list[float], temperatures: list[float], learning_rates: list[float], num_epochs: int = 200,
                     ) -> dict:
    """
    Tune the base CatVAE using grid search over multiple hyperparameter combinations.

    Args:
        train_data:         Training data.
        valid_data:         Validation data.
        train_states:       Ground-truth states for the training data.
        valid_states:       Ground-truth states for the validation data.
        seeds:              Random seeds used for each configuration.
        input_dim:          Input dimension of the data.
        enc_out_dims:       Candidate encoder output dimensions.
        dec_out_dims:       Candidate decoder output dimensions.
        cat_dims:           Candidate categorical latent dimensions.
        betas:              Candidate KL weights.
        temperatures:       Candidate Gumbel-Softmax temperatures.
        learning_rates:     Candidate learning rates.
        num_epochs:         Number of training epochs per run.

    Returns:
        Dictionary containing all tuning results and the best configuration.
    """
    all_results = []
    best_result = None
    best_score = None

    search_space = list(
        product(
            enc_out_dims,
            dec_out_dims,
            cat_dims,
            betas,
            temperatures,
            learning_rates,
        )
    )

    print(f"Starting CatVAE grid search with {len(search_space)} configurations.")

    for config_idx, (
        enc_out_dim,
        dec_out_dim,
        cat_dim,
        beta,
        temperature,
        learning_rate,
    ) in enumerate(search_space, start=1):

        print("=" * 80)
        print(f"Configuration {config_idx}/{len(search_space)}")
        print(
            f"enc_out_dim={enc_out_dim}, "
            f"dec_out_dim={dec_out_dim}, "
            f"cat_dim={cat_dim}, "
            f"beta={beta}, "
            f"temperature={temperature}, "
            f"learning_rate={learning_rate}"
        )

        result = run_base_catvae(
            train_data=train_data,
            valid_data=valid_data,
            train_states=train_states,
            valid_states=valid_states,
            enc_out_dim=enc_out_dim,
            dec_out_dim=dec_out_dim,
            cat_dim=cat_dim,
            beta=beta,
            temperature=temperature,
            seeds=seeds,
            input_dim=input_dim,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )

        summary = result["summary"]
        score_key = f"mean_loss"

        if score_key not in summary:
            raise KeyError(
                f"Objective metric '{score_key}' not found in summary. "
                f"Available keys: {list(summary.keys())}"
            )

        score = summary[score_key]

        result["tuning"] = {
            "config_index": config_idx,
            "objective_metric": "validation loss",
            "objective_score": score,
        }

        all_results.append(result)

        if best_score is None:
            best_score = score
            best_result = result
        else:
            is_better = score < best_score

            if is_better:
                best_score = score
                best_result = result

        print(f"Objective validation loss: {score:.6f}")
        print(f"Best score so far: {best_score:.6f}")

    return {
        "model": "catvae",
        "variant": "base",
        "mode": "tune",
        "objective_metric": "validation loss",
        "search_space": {
            "enc_out_dims": enc_out_dims,
            "dec_out_dims": dec_out_dims,
            "cat_dims": cat_dims,
            "betas": betas,
            "temperatures": temperatures,
            "learning_rates": learning_rates,
            "num_epochs": num_epochs,
            "seeds": seeds,
        },
        "best_result": best_result,
        "all_results": all_results,
    }


def tune_contrastive_catvae(train_data: list, valid_data: list, train_states: list, valid_states: list, seeds: list[int],
                     input_dim: int, enc_out_dims: list[int], dec_out_dims: list[int], cat_dims: list[int],
                     betas: list[float], temperatures: list[float], noises: list[float], info_temperatures: list[float],
                     learning_rates: list[float], num_epochs: int = 200) -> dict:
    """
    Tune the contrastive CatVAE using grid search over multiple hyperparameter combinations.

    Args:
        train_data:         Training data.
        valid_data:         Validation data.
        train_states:       Ground-truth states for the training data.
        valid_states:       Ground-truth states for the validation data.
        seeds:              Random seeds used for each configuration.
        input_dim:          Input dimension of the data.
        enc_out_dims:       Candidate encoder output dimensions.
        dec_out_dims:       Candidate decoder output dimensions.
        cat_dims:           Candidate categorical latent dimensions.
        betas:              Candidate KL weights.
        noises:             The noise for the augmentation.
        info_temperatures:  The parameters for the InfoNCE loss.
        temperatures:       Candidate Gumbel-Softmax temperatures.
        learning_rates:     Candidate learning rates.
        num_epochs:         Number of training epochs per run.

    Returns:
        Dictionary containing all tuning results and the best configuration.
    """
    all_results = []
    best_result = None
    best_score = None

    search_space = list(
        product(
            enc_out_dims,
            dec_out_dims,
            cat_dims,
            betas,
            noises,
            info_temperatures,
            temperatures,
            learning_rates,
        )
    )

    print(f"Starting CatVAE grid search with {len(search_space)} configurations.")

    for config_idx, (
        enc_out_dim,
        dec_out_dim,
        cat_dim,
        beta,
        temperature,
        learning_rate,
        noise,
        info_temperature,
    ) in enumerate(search_space, start=1):

        print("=" * 80)
        print(f"Configuration {config_idx}/{len(search_space)}")
        print(
            f"enc_out_dim={enc_out_dim}, "
            f"dec_out_dim={dec_out_dim}, "
            f"cat_dim={cat_dim}, "
            f"beta={beta}, "
            f"noise={noise}, "
            f"info_temperature={info_temperatures}, "
            f"temperature={temperature}, "
            f"learning_rate={learning_rate}"
        )

        result = run_contrastive_catvae(
            train_data=train_data,
            valid_data=valid_data,
            train_states=train_states,
            valid_states=valid_states,
            noise=noise,
            info_temperature=info_temperature,
            enc_out_dim=enc_out_dim,
            dec_out_dim=dec_out_dim,
            cat_dim=cat_dim,
            beta=beta,
            gumbel_temperature=temperature,
            seeds=seeds,
            input_dim=input_dim,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )

        summary = result["summary"]
        score_key = f"mean_loss"

        if score_key not in summary:
            raise KeyError(
                f"Objective metric '{score_key}' not found in summary. "
                f"Available keys: {list(summary.keys())}"
            )

        score = summary[score_key]

        result["tuning"] = {
            "config_index": config_idx,
            "objective_metric": "validation loss",
            "objective_score": score,
        }

        all_results.append(result)

        if best_score is None:
            best_score = score
            best_result = result
        else:
            is_better = score < best_score

            if is_better:
                best_score = score
                best_result = result

        print(f"Objective validation loss: {score:.6f}")
        print(f"Best score so far: {best_score:.6f}")

    return {
        "model": "catvae",
        "variant": "base",
        "mode": "tune",
        "objective_metric": "validation loss",
        "search_space": {
            "enc_out_dims": enc_out_dims,
            "dec_out_dims": dec_out_dims,
            "cat_dims": cat_dims,
            "betas": betas,
            "temperatures": temperatures,
            "learning_rates": learning_rates,
            "num_epochs": num_epochs,
            "seeds": seeds,
        },
        "best_result": best_result,
        "all_results": all_results,
    }

