import argparse
from benchmark import benchmark_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Huggingface instruct model to benchmark.")
    parser.add_argument('--n_shot', type=int, default=5, help="Number of shots to benchmark. LÄS is kept at 0-shot, due to context length constraints in GPT-SW3 models.")
    args = parser.parse_args()
    benchmark = benchmark_model(args.model, args.n_shot)    

if __name__ == "__main__":
    main()