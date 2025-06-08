import argparse
import subprocess


def run_pretraining():
    print("\n Running Pre-training...")
    subprocess.run(["python", "pre-training.py"])


def run_finetune_sentiment():
    print("\n Running Fine-tuning for Sentiment Analysis...")
    subprocess.run(["python", "fine-tuning_sentiment_analysis.py"])


def run_finetune_topic():
    print("\n Running Fine-tuning for Topic Classification...")
    subprocess.run(["python", "fine-tuning_topic_classification.py"])


def main():
    parser = argparse.ArgumentParser(description="Main script to manage BERT training pipeline")
    parser.add_argument("--stage", choices=["pretrain", "sentiment", "topic", "all"], default="all",
                        help="Stage to run: pretrain, sentiment, topic, or all")
    args = parser.parse_args()

    if args.stage == "pretrain":
        run_pretraining()
    elif args.stage == "sentiment":
        run_finetune_sentiment()
    elif args.stage == "topic":
        run_finetune_topic()
    elif args.stage == "all":
        run_pretraining()
        run_finetune_sentiment()
        run_finetune_topic()


if __name__ == "__main__":
    main()
