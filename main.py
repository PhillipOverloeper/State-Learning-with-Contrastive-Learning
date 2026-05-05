import torch
import argparse

from utils import preprocess_data, split_data
from SOMVAE import run_base_somvae, tune_base_somvae, run_contrastive_somvae, tune_contrastive_somvae
from CATVAE import run_base_catvae, tune_base_catvae, run_contrastive_catvae, tune_contrastive_catvae
from SimCLR import run_base_simclr

# Get the hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the input dimensions of the dataset
DATASET_INPUT_DIMS = {"HAR": 561, "MHEALTH": 23, "PAMAP2": 51}
def get_input_dim(data_name: str) -> int:
    """
    Return the input dimensionality for a supported dataset.
    """
    try:
        return DATASET_INPUT_DIMS[data_name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{data_name}': "
            f"Expected one of: {', '.join(DATASET_INPUT_DIMS)}"
        )


def parse_args():
    """
    Parse command line arguments for training and tuning experiments.

    Usage examples:
        python main.py catvae train base --dataset MHEALTH --seeds 42 43 44
        python main.py catvae tune contrastive --dataset HAR --noise 5e-3 --temperature 0.1
        python main.py somvae train base --dataset PAMAP2
        python main.py somvae tune contrastive --dataset MHEALTH
    """
    parser = argparse.ArgumentParser(
        description="Run experiments for the IEEE CAI paper."
    )

    # 1. Model
    parser.add_argument(
        "--model",
        choices=["somvae", "catvae", "simclr"],
        help="Model family to run.",
    )

    # 2. Mode
    parser.add_argument(
        "--mode",
        choices=["train", "tune"],
        help="Whether to train a fixed configuration or run hyperparameter tuning.",
    )

    # 3. Variant
    parser.add_argument(
        "--variant",
        choices=["base", "contrastive"],
        help="Model variant to run.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_INPUT_DIMS.keys()),
        help="Dataset name.",
    )

    # Seeds
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds, e.g. --seeds 42 43 44.",
    )

    # General training parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio.",
    )

    # SOM-VAE hyperparameters
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument(
        "--som-dim",
        type=int,
        nargs="+",
        default=[2, 2]
    )

    # CatVAE hyperparameters
    parser.add_argument("--enc-out-dim", type=int, default=16)
    parser.add_argument("--dec-out-dim", type=int, default=8)
    parser.add_argument("--cat-dim", type=int, default=20)
    parser.add_argument(
        "--gumbel-temperature",
        type=float,
        default=0.9,
        help="Temperature for the Gumbel-Softmax relaxation.",
    )

    # SimCLR hyperparameters
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--num_states", type=int, default=10)

    # Contrastive hyperparameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for the contrastive InfoNCE loss.",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=5e-3,
        help="Noise level for contrastive augmentation.",
    )

    # Runtime
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use.",
    )

    args = parser.parse_args()
    validate_args(args)

    return args


def validate_args(args):
    """
    Validate model, mode, and variant combinations.
    """
    if args.variant == "contrastive":
        if args.noise <= 0:
            raise ValueError("--noise must be positive for contrastive experiments.")

        if args.temperature <= 0:
            raise ValueError("--temperature must be positive for contrastive experiments.")

    if args.ratio <= 0 or args.ratio >= 1:
        raise ValueError("--ratio must be between 0 and 1.")

    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")


def run_catvae(args: argparse.Namespace, device: torch.device, input_dim: int) -> dict:
    """
    Run CatVAE experiments.

    The execution is controlled by:
        args.mode:    "train" or "tune"
        args.variant: "base" or "contrastive"

    Supported combinations:
        catvae train base
        catvae train contrastive
        catvae tune base
        catvae tune contrastive
    """
    time, data, states, _, _ = preprocess_data(args.dataset, device=device)

    (_, _, train_data, valid_data, train_states, valid_states) = split_data(time, data, states, args.ratio)

    common_metadata = {
        "model": "catvae",
        "variant": args.variant,
        "mode": args.mode,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "train_ratio": args.ratio,
        "seeds": args.seeds,
        "device": str(device),
    }

    if args.mode == "train" and args.variant == "base":
        results = run_base_catvae(train_data, valid_data, train_states, valid_states, args.enc_out_dim, args.dec_out_dim,
                                    args.cat_dim, args.beta, args.gumbel_temperature, args.seeds, input_dim,
                                    args.learning_rate, args.num_epochs)

    elif args.mode == "train" and args.variant == "contrastive":
        results = run_contrastive_catvae(train_data, valid_data, train_states, valid_states, args.noise,
                                         args.temperature, args.seeds, input_dim, args.enc_out_dim, args.dec_out_dim,
                                         args.cat_dim, args.beta, args.gumbel_temperature, args.learning_rate, args.num_epochs)

    elif args.mode == "tune" and args.variant == "base":
        results = tune_base_catvae(train_data, valid_data, train_states, valid_states, args.seeds, input_dim,
                                   enc_out_dims=[4, 8, 16], dec_out_dims=[8, 16, 32], cat_dims=[8, 12, 16, 20],
                                   betas=[0.1, 0.5, 0.8, 1.0], temperatures=[0.5, 0.8, 1.0], learning_rates=[1e-3, 5e-3],
                                   num_epochs=1)


    elif args.mode == "tune" and args.variant == "contrastive":
        results = tune_contrastive_catvae(train_data, valid_data, train_states, valid_states, args.seeds,
                                          input_dim=input_dim, enc_out_dims=[4, 8, 16], dec_out_dims=[8, 16, 32],
                                          cat_dims=[8, 12, 16, 20], betas=[0.1, 0.5, 0.8, 1.0],
                                          temperatures=[0.5, 0.8, 1.0], noises=[1e-3, 1e-4, 1e-5, 1e-6],
                                          info_temperatures=[1e-4, 1e-5, 1e-6], learning_rates=[1e-3, 5e-3],
                                          num_epochs=1)

    else:
        raise ValueError(
            f"Unsupported CatVAE configuration: "
            f"mode={args.mode}, variant={args.variant}"
        )

    return {
        **common_metadata,
        "results": results,
    }


def run_somvae(args: argparse.Namespace, device: torch.device, input_dim: int) -> dict:
    """
    Run SOM-VAE experiments.

    The execution is controlled by:
        args.mode:    "train" or "tune"
        args.variant: "base" or "contrastive"

    Supported combinations:
        somvae train base
        somvae train contrastive
        somvae tune base
        somvae tune contrastive
    """
    time, data, states, _, _ = preprocess_data(args.dataset, device=device)

    (_, _, train_data, valid_data, train_states, valid_states) = split_data(time, data, states, args.ratio)

    common_metadata = {
        "model": "somvae",
        "variant": args.variant,
        "mode": args.mode,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "train_ratio": args.ratio,
        "seeds": args.seeds,
        "device": str(device),
    }

    if args.mode == "train" and args.variant == "base":
        results = run_base_somvae(train_data, valid_data, train_states, valid_states, args.latent,
                                  args.hidden, args.som_dim, args.alpha, args.beta, args.gamma, args.tau,
                                  args.seeds, input_dim, args.learning_rate, args.num_epochs)

    elif args.mode == "train" and args.variant == "contrastive":
        results = run_contrastive_somvae(train_data, valid_data, train_states, valid_states, args.noise,
                                         args.temperature, args.seeds, input_dim, args.latent, args.hidden,
                                         args.som_dim, args.alpha, args.beta, args.gamma,
                                         args.tau, args.learning_rate, args.num_epochs)

    elif args.mode == "tune" and args.variant == "base":
        results = tune_base_somvae(train_data, valid_data, train_states, valid_states, args.seeds, input_dim,
                                   hidden_dims=[4, 8, 16], latent_dims=[8, 16, 32], som_dims=[[2,2], [3,3], [4,4]],
                                   alphas=[0.1, 0.5, 0.8, 1.0], betas=[0.1, 0.5, 0.8, 1.0], gammas=[0.1, 0.5, 0.8, 1.0],
                                   taus=[0.1, 0.5, 0.8, 1.0], learning_rates=[1e-3, 5e-3], num_epochs=1)

    elif args.mode == "tune" and args.variant == "contrastive":
        results = tune_contrastive_somvae(train_data, valid_data, train_states, valid_states, args.seeds,
                                          input_dim=input_dim, hidden_dims=[4, 8, 16], latent_dims=[8, 16, 32],
                                          som_dims=[8, 12, 16, 20], alphas=[0.1, 0.5, 0.8, 1.0],
                                          betas=[0.1, 0.5, 0.8, 1.0], gammas=[0.1, 0.5, 0.8, 1.0],
                                          taus=[0.1, 0.5, 0.8, 1.0], noises=[1e-3, 1e-4, 1e-5, 1e-6],
                                          info_temperatures=[1e-4, 1e-5, 1e-6], learning_rates=[1e-3, 5e-3],
                                          num_epochs=1)

    else:
        raise ValueError(
            f"Unsupported SOM-VAE configuration: "
            f"mode={args.mode}, variant={args.variant}"
        )

    return {
        **common_metadata,
        "results": results,
    }


def run_simclr(args: argparse.Namespace, device: torch.device, input_dim: int) -> dict:
    """
    Run SimCLR experiments.

    The execution is controlled by:
        args.mode:    "train"
        args.variant: "base"

    Supported combinations:
        simclr train base
    """
    time, data, states, _, _ = preprocess_data(args.dataset, device=device)

    (_, _, train_data, valid_data, train_states, valid_states) = split_data(time, data, states, args.ratio)

    common_metadata = {
        "model": "simclr",
        "variant": args.variant,
        "mode": args.mode,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "train_ratio": args.ratio,
        "seeds": args.seeds,
        "device": str(device),
    }

    if args.mode == "train" and args.variant == "base":
        results = run_base_simclr(train_data, valid_data, train_states, valid_states, args.latent_dim,
                                  args.hidden_dim, args.seeds, args.noise, args.temperature, args.num_states,
                                  input_dim, args.learning_rate, args.num_epochs)

    else:
        raise ValueError(
            f"Unsupported SimCLR configuration: "
            f"mode={args.mode}, variant={args.variant}"
        )

    return {
        **common_metadata,
        "results": results,
    }










if __name__ == '__main__':
    args = parse_args()

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Variant: {args.variant}")
    print(f"Seeds: {args.seeds}")

    input_dim = get_input_dim(args.dataset)

    if args.model == "catvae":
        run_catvae(args, device, input_dim)
    elif args.model == "somvae":
        run_somvae(args, device, input_dim)
    elif args.model == "simclr":
        run_simclr(args, device, input_dim)






    """
    #args = parse_seeds()
    #print(args)

    #args = parse_node_args()
    #seeds = list(map(int, args.seeds.split()))

    dataset = 'MHEALTH'

    seeds = [42]

    #get_fode_hyperparameter_results(seeds, dataset, args)

    alpha = 1
    beta = 1
    gamma = 0.8
    tau = 0.8
    hidden_dim = 16
    latent_dim = 16
    # seeds = [1]
    # tune_base_som(latent, hidden, alpha, beta, gamma, tau, seeds)
    # tune_base_som(args.latent, args.hidden, args.alpha, args.beta, args.gamma, args.tau, seeds, dataset)

    #seeds = [42, 43]
    # get_best_som_results(seeds, dataset)

    #enc_out_dim = 16
    #dec_out_dim = 8
    #cat_dim = 20
    #beta = 0.7

    #seeds = [42, 43]
    #temperature = 0.001
    #noise = 1e-1
    #print('Noise:', args.noise, 'Temp: ', args.temperature)
    #tune_contrastive_som(args.noise, args.temperature, seeds, dataset)
    #temp = 0.9
    #seeds = [42]

    #tune_base_cat(args.enc_out_dim, args.dec_out_dim, args.cat_dim, args.beta, args.temp, seeds, dataset)


    #noise = 1e-6
    #print('Noise:', noise, 'Temp: ', args.temperature)

    #seeds = [42, 43]
    #tune_contrastive_cat(noise, args.temperature, seeds, dataset)
    #get_best_cat_results(seeds, dataset)

    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52]
    temperature = 1e-8
    noise = 5e-3
    print('Noise:', noise, 'Temp: ', temperature)
    get_simclr_results(noise, temperature, seeds, dataset)"""




