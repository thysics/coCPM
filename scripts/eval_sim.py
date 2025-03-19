import argparse
from workflow.evaluate import Eval



def main():
    parser = argparse.ArgumentParser(description="Run evaluation script")
    
    # Experiment
    parser.add_argument("--experiment_mode", type=str, default="train", help="To run experiment in training or eval mode")
    
    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--train_data_star", type=str, required=True, help="Path to simulated OOD training data CSV")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data CSV")
    
    # Hyperparams
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs")            # Note: models arent't converging with only 300
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for evaluation")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 1.0], help="List of lambda values",)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation",)
    parser.add_argument("--n_sample", type=int, default=50, help="Number of samples for evaluation")

    args = parser.parse_args()
    
    evaluator = Eval(
        args.train_data, args.train_data_star, args.test_data, lr=0.001, hidden_dim=8              # Increased learning rate
    )
    
    if args.experiment_mode.lower() == "train":
        evaluator.train(
            batch_size=args.batch_size,
            lambdas=args.lambdas,
            n_epochs=args.n_epochs,
            max_wait=10,                                                                            # Updated. 40 validation intervals at 10 epochs apart meant max wait was 400 epochs
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
