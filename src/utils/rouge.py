from collections import defaultdict
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import numpy as np

import logging
import nltk
import json
import os
import re

import sys

sys.setrecursionlimit(1000000000)

logging.getLogger().setLevel(logging.INFO)


def rouge_pipeline(dataset_path, split, reports_folder, summaries_folder, destination_path, rouge):
    """Extract the matching summaries associated with a given report and extract the labels for
    the current split

    :param dataset_path: (str) root path of the full dataset
    :param split: (str) split of the dataset of interest (train,val,test)
    :param reports_folder: (str) name of the folder containing the reports
    :param summaries_folder: (str) name of the folder containing the summaries
    :param destination_path: (str) name of the folder where the w2v model will be stored
    :param rouge: (Rouge) object that calculates the rouge-l score given a pair of (report,summary)
    :return: void
    """
    REPORTS_PATH = os.path.join(dataset_path, split, reports_folder)
    SUMMARIES_PATH = os.path.join(dataset_path, split, summaries_folder)
    DESTINATION_PATH = os.path.join(destination_path, split)

    if not os.path.exists(DESTINATION_PATH):
        os.makedirs(DESTINATION_PATH)

    logging.info(f"[{split}] Extracting matching summaries for the given reports...")
    matching_summaries = get_matching_summaries(REPORTS_PATH, SUMMARIES_PATH)

    logging.info(f"[{split}] Extracting summary labels based on Rouge-L score...")
    for report_path in tqdm(list(matching_summaries.keys())):
        create_label(rouge, report_path, matching_summaries[report_path], DESTINATION_PATH)


def get_matching_summaries(reports_path, summaries_path):
    """It is needed to match the sentences of the annual report
    with the ones in the golden summary folder. It verifies the id and
    creates a dictionary in which we can find as key the name of the
    file and as values the list of all the associated summaries.

    :param reports_path: (str) path of all the given reports for the current splitted dataset (either train, val or test)
    :param summaries_path: (str) path of all the given summaries for the current splitted dataset
    :return: (dict) dictionary containing the matches between the annual reports and its summaries.
            e.g. {'../../10023.txt' : ['../../10023_1.txt','../../10023_2.txt','../../10023_3.txt'...]}
    """

    dataset_index = defaultdict(list)
    # add each summary to its report array
    for summary in os.listdir(summaries_path):

        id = summary.split("_")[0] + '.txt' if '_' in summary else summary
        path_id = os.path.join(reports_path, id)
        dataset_index[path_id].append(os.path.join(summaries_path, summary))

    return dataset_index


def create_label(rouge, report_path, summaries_path, destination_path):
    """Create the given summary label for each report. This will be chosen
    according to the ROUGE-L score across all its provided summary, considering
    the maximum one.
    The dataset will rely on a set of json files, one for each record, where we'll have
            the following keys:
                - report : report corpys,
                - summary : full summary of the summary having the highest rouge,
                - label_sentence : extracted sentence from the corpus having the highest rouge score,
                - rouge_score : maximum rouge score obtained
    :param rouge: (Rouge) object that calculates the rouge-l score given a pair of (report,summary)
    :param report_path: (str) path of all the given reports for the current splitted dataset (either train,val or test)
    :param summaries_path: (str) path of all the given summaries for the current splitted dataset
    :param destination_path: (str) path of the directory where the new dataset will be stored
    :return: void
    """

    # initialize dictionary that will contain our dataset information
    curr_data = {
        'report': None,
        'summary': None,
        'extracted': [],
        'score': []
    }

    # read report
    with open(report_path, 'r', encoding="utf-8") as f_report:
        curr_data['report'] = f_report.readlines()
        f_report.close()

    for summary_path in summaries_path:

        with open(summary_path, 'r', encoding="utf-8") as f_summary:
            curr_data['summary'] = f_summary.readlines()

        # the rouge-l will be calculated considering the most similar
        # sentence of the report for each sentence of the summary
        for idx, sentence in enumerate(curr_data['summary']):

            # print(sentence)

            # remove special chars
            sentence = re.sub(r" <EOS> ", "", sentence)
            sentence = re.sub(r" <SOS> ", "", sentence)
            sentence = re.sub(r"\n", "", sentence)

            # extract the rouge score of the most similar sentence of the report with respect
            # to the summary.

            # extracted_sent will be the idx of the selected report summary ?
            rouge_scores, extracted_sents = get_rouge_score(curr_data['report'], sentence, rouge, 5)

            curr_data['extracted'].extend(extracted_sents)
            curr_data['score'].extend(rouge_scores)

        # dump the best results on the provided directory
        summary_filepath = summary_path.split("/")[-1].split(".")[0].split("_")[0] # remove '_'

        curr_data['extracted'] = curr_data['extracted']
        curr_data['score'] = curr_data['score']

        with open(os.path.join(destination_path, f'{summary_filepath}.json'), 'w') as f:
            json.dump(curr_data, f, indent=4)


def get_rouge_score(report, summary_sentence, rouge, n_sentences=1):
    """Compute the rouge-l score for a given sentence of the summary
    with respect to all the sentences on the report
    :param sentence: (str) sentence extracted from the full report
    :param summaries: (list) list of golden summaries associated to the report
    :param rouge: (Rouge) object that calculates the rouge-l score given a pair of (report,summary)
    :return: best summary for the given sentence and its maximum rouge score
    """

    scores = []
    sentence_ids = []

    for idx, report_sentence in enumerate(report):

        report_sentence = re.sub(r" <EOS> ", "", report_sentence)
        report_sentence = re.sub(r" <SOS> ", "", report_sentence)
        report_sentence = re.sub(r"\n", "", report_sentence)

        if len(report_sentence) < 10:
            continue

        try:
            score_rouge = rouge.get_scores(report_sentence, summary_sentence)


        except Exception as e:
            print(str(e))
            print("Summary sentence = ", summary_sentence)
            print("=========")
            print('Report sentence = ', report_sentence)

        scores.append(score_rouge[0]["rouge-l"]["f"])
        sentence_ids.append(idx)

    keep_indices = [i[0] for i in sorted(enumerate(scores), key=lambda k: k[1], reverse=True)][:n_sentences]

    final_scores = [round(scores[idx],2) for idx in keep_indices]
    final_ids = [sentence_ids[idx] for idx in keep_indices]

    return final_scores, final_ids