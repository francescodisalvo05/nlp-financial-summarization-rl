import logging
import string
import nltk
import ftfy
import json
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

from tqdm import tqdm

import numpy as np

logging.getLogger().setLevel(logging.INFO)


def clean_pipeline(dataset_path, reports_folder, summaries_folder, preprocessed_path, corpus_filename, split, nlp):
    """Setup the directories, preprocess the text and save the cleaned ones at <preprocessed_path>. Moreover, for
    training reports, the full corpus will be saved in order to train w2v.
    :param dataset_path: (str) path of the root dataset
    :param reports_folder: (str) name of the folder containing the reports
    :param summaries_folder: (str) name of the folder containing the summaries
    :param preprocessed_path: (str) path of the folder that will store the preprocessed path
    :param corpus_filename: (str) filename of the corpus used later for training w2v
    :param split: (str) dataset split (train,val,test)
    :param nlp: (spacy)
    :return: void
    """
    logging.info(f"Setting up {split} cleaning directories...")
    PATH_CORPUS = os.path.join(preprocessed_path, corpus_filename) if corpus_filename else None
    PATH_REPORTS = os.path.join(dataset_path, split, reports_folder)
    PATH_SUMMARIES = os.path.join(dataset_path, split, summaries_folder)
    PATH_REPORTS_PREPROCESSED = os.path.join(preprocessed_path, split, reports_folder)
    PATH_SUMMARIES_PREPROCESSED = os.path.join(preprocessed_path, split, summaries_folder)

    if not os.path.exists(PATH_REPORTS_PREPROCESSED):
        os.makedirs(PATH_REPORTS_PREPROCESSED)
    if not os.path.exists(PATH_SUMMARIES_PREPROCESSED):
        os.makedirs(PATH_SUMMARIES_PREPROCESSED)

    logging.info(f"Cleaning {split} input reports...")
    clean_files(PATH_REPORTS, PATH_REPORTS_PREPROCESSED, PATH_CORPUS, nlp)

    logging.info(f"Cleaning {split} input gold summaries...")
    clean_files(PATH_SUMMARIES, PATH_SUMMARIES_PREPROCESSED, PATH_CORPUS, nlp)


def filter_pipeline(preprocessed_path, reports_folder, summaries_folder, filtered_path, corpus_filename,
                    filtered_corpus_filename, tokens_to_filter_filepath, split):
    logging.info(f"Setting up {split} filtering directories...")

    PATH_CORPUS = os.path.join(preprocessed_path, corpus_filename) if corpus_filename else None
    PATH_FILTERED_CORPUS = os.path.join(filtered_path, filtered_corpus_filename) if filtered_corpus_filename else None
    PATH_FILTERED_TOKENS = os.path.join(filtered_path, tokens_to_filter_filepath) if tokens_to_filter_filepath else None

    PATH_REPORTS = os.path.join(preprocessed_path, split, reports_folder)
    PATH_SUMMARIES = os.path.join(preprocessed_path, split, summaries_folder)
    PATH_REPORTS_FILTERED = os.path.join(filtered_path, split, reports_folder)
    PATH_SUMMARIES_FILTERED = os.path.join(filtered_path, split, summaries_folder)

    tokens_to_filter = get_tokens_to_filter(PATH_CORPUS, PATH_FILTERED_TOKENS, split)

    logging.info(f"Filtering {split} input reports...")
    filter_tokens(PATH_REPORTS, PATH_REPORTS_FILTERED, PATH_FILTERED_CORPUS, tokens_to_filter)

    logging.info(f"Filtering {split} input gold summaries...")
    filter_tokens(PATH_SUMMARIES, PATH_SUMMARIES_FILTERED, PATH_FILTERED_CORPUS, tokens_to_filter)

    return tokens_to_filter


def clean_files(input_path, destination_path, corpus_path, nlp):
    """
    Read all the files in the input directory and clean them all sentence by sentence.
    Then, appen the full corpus at <corpus_path> and store the cleaned sentences to
    <destination_path>.
    :param input_path: (str) path of the directory containing all the files that must be cleaned
    :param destination_path: (str) path of the directory where all the preprocessed files will be stored
    :param corpus_path: (str) path of the full corpus to update. For the validation set, test set and golden summaries it
                                 will be None, therefore do not write
    :param nlp: (spacy)
    :return:
    """

    list_files = os.listdir(input_path)

    for filename in tqdm(sorted(list_files)):

        if filename.split(".")[-1] != "txt":
            continue

        filepath = os.path.join(input_path, filename)

        report_file = open(filepath, "r", encoding="utf-8").read()
        report_file = ftfy.fix_text(report_file)  # fix text through encoding
        report_file = report_file.replace('\n', '')  # remove \n
        report_file = report_file.replace('\t', ' ')  # remove \t

        '''
        # Spacy's max length
        if len(report_file) > 150000:
            report_file = report_file[:150000]
        '''

        sentences = sent_tokenize(report_file)

        if not os.path.exists(destination_path):
            os.makedirs(destination_path, mode=0o666)

        with open(os.path.join(destination_path, filename), 'w+', encoding="utf8") as destination_file:

            for sentence in sentences:

                cleaned_sentence = clean_sentence(sentence)

                if len(word_tokenize(cleaned_sentence)) > 60:
                    cleaned_sentence = ' '.join(word_tokenize(cleaned_sentence[:60]))

                if len(cleaned_sentence) > 1:
                    # add Start of Sentence and End of Sentence special tokens
                    cleaned_sentence_special_chars = ' <SOS> ' + cleaned_sentence + ' <EOS> \n'

                    # update the preprocessed file
                    destination_file.write(cleaned_sentence_special_chars)

                    if corpus_path:
                        with open(corpus_path, 'a+', encoding="utf8") as corpus_file:
                            # update the corpus file, if this file is from
                            corpus_file.write(cleaned_sentence_special_chars)


def get_tokens_to_filter(corpus_path, PATH_FILTERED_TOKENS, split):

    if split == 'train':

        tokens_list = []  # list of lists

        with open(corpus_path, 'r') as f:
            for line in f.readlines():
                line = line.replace(' <SOS> ', '')
                line = line.replace(' <EOS> ', '')
                tokens_list.append(word_tokenize(line))
            f.close()

        corpus_tokens = [token for sentence in tokens_list for token in sentence]
        bow = nltk.FreqDist(corpus_tokens)

        tokens_to_filter = dict(bow.most_common(20000))
        
        if not '<SOS>' in tokens_to_filter.keys():
            tokens_to_filter['<SOS>'] = -1
        if not '<EOS>' in tokens_to_filter.keys():
            tokens_to_filter['<EOS>'] = -1

        with open(PATH_FILTERED_TOKENS, "w") as outfile:
            json.dump(tokens_to_filter, outfile, indent=4)


        return tokens_to_filter


    else:

        with open(PATH_FILTERED_TOKENS, 'r') as f:
            tokens_to_filter = json.load(f)

        return tokens_to_filter


def filter_tokens(input_path, destination_path, filtered_corpus_path, tokens_to_filter):
    list_files = os.listdir(input_path)

    if not os.path.exists(destination_path):
        os.makedirs(destination_path, mode=0o666)

    for filename in tqdm(list_files):

        with open(os.path.join(input_path, filename)) as curr_file:
            sentences = curr_file.readlines()

            with open(os.path.join(destination_path, filename), 'w+', encoding="utf8") as destination_file:

                for sentence in sentences:

                    if len(sentence) > 1:
                        # update the preprocessed file
                        text = ' '.join([tk for tk in sentence.split(' ') if tk in tokens_to_filter])

                        if text == ' <SOS> <EOS> \n':
                            continue

                        destination_file.write(text + '\n')

                        # training set
                        if filtered_corpus_path:
                            with open(filtered_corpus_path, 'a', encoding="utf8") as corpus_file:
                                # update the corpus file, if this file is from
                                corpus_file.write(text + '\n')


def clean_sentence(sentence):
    """
    Clean sentences: remove non alpha numerical values, update rounded money
    values (100 M -> 100 million), normalize, remove links and split numbers and
    strings that are often adjacents.
    :param sentence: (str) input sentence that needs to be cleaned
    :return: (str)  cleaned sentence
    """

    sentence = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",
                      sentence)  # remove non alpha numerical values
    sentence = re.sub(r"[m/M]\b", " million", sentence)  # replace m or M with Million when following a number
    sentence = re.sub(r"\b[m/M]\b", "million", sentence)  # replace m or M with Million when standalone letter
    sentence = re.sub(r"[b/B]\b", " billion", sentence)  # replace m or M with Million when following a number
    sentence = re.sub(r"\b[b/B]\b", "billion", sentence)  # replace m or M with Million when standalone letter
    sentence = sentence.lower()

    # replace url with text python : https://stackoverflow.com/questions/21932615/regular-expression-for-remove-link
    sentence = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', sentence, flags=re.MULTILINE)

    # split number and strings  for example: foo123 or 123foo
    sentence = split_numbers_strings(sentence)
    return sentence


def split_numbers_strings(sentence):
    """Split adjacents numbers with strings, e.g. 12345xxxx. This was
    a regular pattern after sentence and word tokenization.
    :param sentence: (str) input sentence that needs to be cleaned
    :return: (str)  cleaned sentence
    """

    sentence = sentence.split()
    cleaned_sentence = []

    for token in sentence:

        t = token

        match_1 = re.match(r"([a-z]+)([0-9]+)", token, re.I)
        if match_1:
            items_1 = match_1.groups()
            t = ' '.join(items_1)

        match_2 = re.match(r"([0-9]+)([a-z]+)", token, re.I)
        if match_2:
            items_2 = match_2.groups()
            t = ' '.join(items_2)

        cleaned_sentence.append(t)

    return ' '.join(cleaned_sentence)