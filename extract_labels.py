"""Extract labels for each report. The label with
be the sentence having the highest ROUGE-L score across
all the provided gold summaries
"""

import argparse
import os

from rouge import Rouge
from evaluate import load
from tqdm import tqdm

from utils.rouge import rouge_pipeline
from utils.bleu import bleu_pipeline
from utils.bert_score import bert_pipeline



def main(args):

    rouge = Rouge(metrics=['rouge-l'])
    bertscore = load("bertscore")

    # # # set directories
    train_flag = 'train' in args.dataset_split
    val_flag = 'val' in args.dataset_split
    test_flag = 'test' in args.dataset_split

    if not os.path.exists(args.destination_path):
        os.makedirs(args.destination_path)

    if args.metric == "rouge":
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
    

    elif args.metric == "bert":
    # # # create labels based on the bert score for each split
    # maximum bert score recall among each pair of sentence for all the
    # provided golden summaries

        if train_flag:
            bert_pipeline(args.dataset_path, 'train', args.reports_folder,
                        args.summaries_folder, args.destination_path,bertscore)

        if val_flag:
            bert_pipeline(args.dataset_path, 'val', args.reports_folder,
                        args.summaries_folder, args.destination_path,bertscore)

        if test_flag:
            bert_pipeline(args.dataset_path, 'test', args.reports_folder,
                        args.summaries_folder, args.destination_path,bertscore)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dataset_path', type=str, required=True, help='Set the base dataset path')
    parser.add_argument('-P', '--destination_path', type=str, required=True, help='Set the path of the final labeled dataset')
    parser.add_argument('-d', '--dataset_split', nargs='+', required=True,
                        help='Set the dataset split to preprocess (even more than one)',
                        choices=['train', 'val', 'test'])

    parser.add_argument('-r', '--reports_folder', type=str, default='annual_reports', help='Name of the reports folder')
    parser.add_argument('-s', '--summaries_folder', type=str, default='gold_summaries', help='Name of the summaries folder')
    parser.add_argument('-m', '--metric', type=str, required=True, default='rouge', choices=['rouge', 'bleu','bert'],help='Metric to be used')


    args = parser.parse_args()

    main(args)