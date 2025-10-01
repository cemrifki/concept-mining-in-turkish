import argparse

from src import extracting_concepts_from_docs as ecd, generating_turkish_hypernyms as gth


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process Turkish text documents recursively for summarization and concept extraction."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset folder containing documents (possibly in subfolders).",
    )
    parser.add_argument(
        "--lexical-db",
        type=str,
        required=True,
        help="Path to Turkish_words_and_hypernyms.csv file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Folder to save output JSON files (mirrors dataset structure).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top sentences to extract.",
    )
    parser.add_argument(
        "--generate-hypernyms",
        action="store_true",
        help="If set, regenerate the Turkish hypernym dictionary prior to concept extraction.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_hypernyms:
        gth.main(args)
    ecd.main(args)

"""
Example usages:

python main.py \
  --dataset datasets \
  --lexical-db resources/lexical_database/Turkish_words_and_hypernyms.csv \
  --output output \
  --top-k 3

python main.py \
  --dataset ./datasets \
  --lexical-db ./resources/lexical_database/Turkish_words_and_hypernyms.csv \
  --output ./output \
  --top-k 5 \
  --generate-hypernyms

"""
