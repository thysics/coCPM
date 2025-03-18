import argparse
from workflow.evaluate import (
    Eval,
)  # Assuming the original script is saved as eval_script.py


def main():
    parser = argparse.ArgumentParser(description="Run evaluation script")
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data CSV"
    )
    parser.add_argument(
        "--train_data_star",
        type=str,
        required=True,
        help="Path to simulated OOD training data CSV",
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data CSV"
    )
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations for evaluation"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.1, 1.0],
        help="List of lambda values",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--n_sample", type=int, default=50, help="Number of samples for evaluation"
    )

    args = parser.parse_args()

    evaluator = Eval(args.train_data, args.train_data_star, args.test_data)
    evaluator.train(
        batch_size=args.batch_size,
        lambdas=args.lambdas,
        hidden_dim=8,
        lr=0.001,
        n_epochs=args.n_epochs,
        max_wait=40,
    )
    results = evaluator.evaluate(
        iterations=args.iterations,
        lambdas=args.lambdas,
        batch_size=args.batch_size,
        n_sample=args.n_sample,
    )

    print("Evaluation Results:")
    print(results)


if __name__ == "__main__":
    main()
