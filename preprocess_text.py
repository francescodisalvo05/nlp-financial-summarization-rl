from utils.preprocessing import clean_pipeline

import argparse
import logging
import spacy
import os


def main(args):

    # # # set SPACY
    logging.info("Setting up SPACY...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 150000

    # # # set directories and preprocess the text
    train_flag = 'train' in args.dataset_split
    val_flag = 'val' in args.dataset_split
    test_flag = 'test' in args.dataset_split

    if not os.path.exists(args.preprocessed_path):
        os.makedirs(args.preprocessed_path)


    if train_flag:
        clean_pipeline(args.dataset_path, args.reports_folder,
                       args.summaries_folder, args.preprocessed_path,
                       'full_corpus.txt', 'train',nlp)

    if val_flag:
        clean_pipeline(args.dataset_path, args.reports_folder,
                       args.summaries_folder, args.preprocessed_path,
                       None,'val',nlp)

    if test_flag:
        clean_pipeline(args.dataset_path, args.reports_folder,
                       args.summaries_folder, args.preprocessed_path,
                       None, 'test', nlp)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dataset_path', type=str, required=True, help='Set the base dataset path')
    parser.add_argument('-P', '--preprocessed_path', type=str, required=True, help='Set the base dataset path for the preprocessed files')

    parser.add_argument('-d', '--dataset_split', nargs='+', required=True, help='Set the dataset split to preprocess (even more than one)',
                        choices=['train','val','test'])

    parser.add_argument('-r', '--reports_folder', type=str, required=True, help='Name of the reports folder')
    parser.add_argument('-s', '--summaries_folder', type=str, required=True, help='Name of the summaries folder')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)