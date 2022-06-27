"""Extract labels for each report. The label with
be the sentence having the highest ROUGE-L score across
all the provided gold summaries
"""

import argparse
import os

from rouge import Rouge
from tqdm import tqdm

from utils.rouge import rouge_pipeline


def main(args):

    rouge = Rouge(metrics=['rouge-l'])

    # # # set directories
    train_flag = 'train' in args.dataset_split
    val_flag = 'val' in args.dataset_split
    test_flag = 'test' in args.dataset_split

    if not os.path.exists(args.destination_path):
        os.makedirs(args.destination_path)


    # # # create labels based on the rouge score for each split
    # maximum rouge-l score among each pair of sentence for all the
    # provided golden summaries

    if train_flag:
        rouge_pipeline(args.dataset_path, 'train', args.reports_folder,
                       args.summaries_folder, args.destination_path, rouge)

    if val_flag:
        rouge_pipeline(args.dataset_path, 'val', args.reports_folder,
                       args.summaries_folder, args.destination_path, rouge)

    if test_flag:
        rouge_pipeline(args.dataset_path, 'test', args.reports_folder,
                       args.summaries_folder, args.destination_path, rouge)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dataset_path', type=str, required=True, help='Set the base dataset path')
    parser.add_argument('-P', '--destination_path', type=str, required=True, help='Set the path of the final labeled dataset')
    parser.add_argument('-d', '--dataset_split', nargs='+', required=True,
                        help='Set the dataset split to preprocess (even more than one)',
                        choices=['train', 'val', 'test'])

    parser.add_argument('-r', '--reports_folder', type=str, default='annual_reports', help='Name of the reports folder')
    parser.add_argument('-s', '--summaries_folder', type=str, default='gold_summaries', help='Name of the summaries folder')

    args = parser.parse_args()

    main(args)