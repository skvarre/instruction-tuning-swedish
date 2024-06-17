import argparse
from benchmark import benchmark_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Huggingface instruct model to benchmark.")
    args = parser.parse_args()
    
    benchmark = benchmark_model(args.model)    

if __name__ == "__main__":
    main()