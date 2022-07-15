from src.utils.preprocessing import clean_pipeline, filter_pipeline

import argparse
import logging
import os


def main(args):

    if not os.path.exists(args.preprocessed_path):
        os.makedirs(args.preprocessed_path)

    if not os.path.exists(args.filtered_path):
        os.makedirs(args.filtered_path)

    # train
    clean_pipeline(args.dataset_path, args.reports_folder,
                   args.summaries_folder, args.preprocessed_path,
                   'full_corpus.txt', 'train')

    filter_pipeline(args.preprocessed_path, args.reports_folder,
                       args.summaries_folder, args.filtered_path,
                       'full_corpus.txt', 'full_corpus_filtered.txt',
                       'tokens_to_remove.json', 'train')

    # validation
    clean_pipeline(args.dataset_path, args.reports_folder,
                   args.summaries_folder, args.preprocessed_path,
                   None, 'val')

    filter_pipeline( args.preprocessed_path, args.reports_folder,
                     args.summaries_folder, args.filtered_path,
                     'full_corpus.txt', 'full_corpus_filtered.txt',
                     'tokens_to_remove.json', 'val')

    # test
    clean_pipeline(args.dataset_path, args.reports_folder,
                   args.summaries_folder, args.preprocessed_path,
                   None, 'test')

    filter_pipeline(args.preprocessed_path, args.reports_folder,
                        args.summaries_folder, args.filtered_path,
                        'full_corpus.txt', 'full_corpus_filtered.txt',
                        'tokens_to_remove.json', 'test')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dataset_path', type=str, required=True, help='Set the base dataset path')
    parser.add_argument('-P', '--preprocessed_path', type=str, required=True, help='Set the base dataset path for the preprocessed files')
    parser.add_argument('-F', '--filtered_path', type=str, required=True,
                        help='Set the base dataset path for the filtered files')

    parser.add_argument('-r', '--reports_folder', type=str, required=True, help='Name of the reports folder')
    parser.add_argument('-s', '--summaries_folder', type=str, required=True, help='Name of the summaries folder')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)

