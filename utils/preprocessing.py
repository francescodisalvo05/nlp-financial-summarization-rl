import logging
import string
import nltk
import ftfy
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

from tqdm import tqdm

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
    logging.info(f"Setting up {split} directories...")
    PATH_CORPUS =  os.path.join(preprocessed_path, corpus_filename) if corpus_filename else None
    PATH_REPORTS = os.path.join(dataset_path, split, reports_folder)
    PATH_SUMMARIES = os.path.join(dataset_path, split, summaries_folder)
    PATH_REPORTS_PREPROCESSED = os.path.join(preprocessed_path, split, reports_folder)
    PATH_SUMMARIES_PREPROCESSED = os.path.join(preprocessed_path, split, summaries_folder)

    logging.info(f"Cleaning {split} input reports...")
    clean_files(PATH_REPORTS, PATH_REPORTS_PREPROCESSED,PATH_CORPUS, nlp)

    logging.info(f"Cleaning {split} input gold summaries...")
    clean_files(PATH_SUMMARIES, PATH_SUMMARIES_PREPROCESSED,PATH_CORPUS, nlp)




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

    for filename in tqdm(list_files):

        filepath = os.path.join(input_path,filename)
        report_file = open(filepath, "r", encoding="utf-8").read()
        report_file = ftfy.fix_text(report_file)  # fix text through encoding
        report_file = report_file.replace('\n', '')  # remove \n
        report_file = report_file.replace('\t', ' ')  # remove \t

        # Spacy's max length
        if len(report_file) > 150000:
            report_file = report_file[:150000]

        sentences = nlp(report_file)

        if not os.path.exists(destination_path):
          os.makedirs(destination_path,mode=0o666)

        with open(os.path.join(destination_path,filename), 'w+', encoding="utf8") as destination_file:

            for sentence in sentences.sents:
                cleaned_sentence = clean_sentence(sentence.text)
                if len(cleaned_sentence) > 1:
                    # add Start of Sentence and End of Sentence special tokens
                    cleaned_sentence_special_chars = ' <SOS> ' + cleaned_sentence + ' <EOS> \n'

                    # update the preprocessed file
                    destination_file.write(cleaned_sentence_special_chars)

                    if corpus_path:
                        with open(corpus_path, 'a', encoding="utf8") as corpus_file:
                            # update the corpus file, if this file is from
                            corpus_file.write(cleaned_sentence_special_chars)


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