from gensim.models import Word2Vec

import argparse
import logging
import sys


def train_w2v(args):

    logging.info(f"Extracting the full corpus from '{args.corpus_path}'...")
    corpus = []
    with open(args.corpus_path, "r", encoding="utf8") as f:
        for line in f.readlines():
            corpus.append(line.strip().split(" "))

    logging.info(f"Training w2v...")
    start = time()
    model = Word2Vec(corpus_filtered,
                     sg=1, min_count=3, window=args.window,
                     vector_size=args.vector_size, sample=6e-5, alpha=0.05,
                     negative=20, workers=16, epochs=15)
    logging.info(f"Training completed in {timedelta(seconds=time()-start)}...")

    model.save(PATH_W2V)
    logging.info(f"Model saved at '{args.destination_path}'")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # directory settings
    parser.add_argument('-c','--corpus_path', required=True, help='Path of the full corpus')
    parser.add_argument('-d','--destination_path', required=True, help='Destination path of the w2c model')

    # w2v hyperaparameters
    parser.add_argument('--vector_size', type=int, default=300, help='Vector size of w2v')
    parser.add_argument('--negative', type=int, default=20, help='Number of negative samples for w2v')
    parser.add_argument('--window', type=int, default=2, help='Window size of w2v')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs of w2v')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    train_w2v(args)