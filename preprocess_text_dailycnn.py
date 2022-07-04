from datasets import load_dataset

from utils.preprocessing_daily import clean_pipeline

import logging

import argparse


def main(args):

    # object containing 'train', 'val', 'test' datasets
    full_dataset = load_dataset("cnn_dailymail",'3.0.0')

    # CLEAN DATA
    if 'train' in args.dataset_split:
        clean_pipeline(full_dataset['train'], args.preprocessed_path,'full_corpus.txt', 'train')

    if 'val' in args.dataset_split:
        clean_pipeline(full_dataset['validation'], args.preprocessed_path, None, 'val')

    if 'test' in args.dataset_split:
        clean_pipeline(full_dataset['test'], args.preprocessed_path, None, 'test')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-P', '--preprocessed_path', type=str, required=True,
                        help='Set the base dataset path for the preprocessed files')
    parser.add_argument('-F', '--filtered_path', type=str, required=True,
                        help='Set the base dataset path for the filtered files')

    parser.add_argument('-d', '--dataset_split', nargs='+', required=True,
                        help='Set the dataset split to preprocess (even more than one)',
                        choices=['train', 'val', 'test'])


    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)


    main(args)