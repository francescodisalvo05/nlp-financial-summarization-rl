from nltk.tokenize import word_tokenize, sent_tokenize

from tqdm import tqdm

import logging
import random
import json
import nltk
import ftfy
import re
import os


def clean_pipeline(dataset, destination_path, corpus_filename, split):

    # setup root directory
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # create train/val/test directories
    if not os.path.exists(os.path.join(destination_path,split)):
        os.makedirs(os.path.join(destination_path,split))

    # follow FNS data format
    # create subdirectories [texts, summaries]
    if not os.path.exists(os.path.join(destination_path,split,'texts')):
        os.makedirs(os.path.join(destination_path,split,'texts'))

    if not os.path.exists(os.path.join(destination_path,split,'summaries')):
        os.makedirs(os.path.join(destination_path,split,'summaries'))


    corpus_path = os.path.join(destination_path,corpus_filename) if corpus_filename else None

    logging.info(f'Cleaning {split} data...')

    sample_documents = {
        'train' : 25000,
        'val' : 1000,
        'test' : 1000
    }

    # sample without replacement
    random_ids = random.sample(range(len(dataset)), sample_documents[split])

    # save ids
    with open(os.path.join(destination_path,f'{split}_ids.txt'),'w') as id_file:

        for idx in tqdm(random_ids):

            id_file.write(f'{str(idx)}\n')

            # use the same notation as FNS dataset
            # store them in the same format too
            # -- dataset root
            # | ---- texts
            # | ---- summaries

            text = preprocess_text(dataset['article'][idx], corpus_path, max_tokens=800)
            summary = preprocess_text(dataset['highlights'][idx], corpus_path, max_tokens=100)

            # save text
            with open(os.path.join(destination_path,split,'texts',f'{str(idx+1).zfill(2)}.txt'), 'w+') as t:
                for text_sentence in text:
                    t.write(text_sentence)

            # save summary
            with open(os.path.join(destination_path, split, 'summaries', f'{str(idx + 1).zfill(2)}.txt'), 'w+') as s:
                for summary_sentence in summary:
                    s.write(summary_sentence)


def preprocess_text(text, corpus_path, max_tokens):

    sentences = sent_tokenize(text)
    cleaned_sentences = []
    word_counter = 0

    for sentence in sentences:

        sentence = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", sentence)
        sentence = sentence.lower()

        # cast at 800 words for full text and 100 for the target
        if word_counter >= max_tokens:
            break

        tokenized_sent = word_tokenize(sentence)

        # cast sentence
        if word_counter + len(tokenized_sent) >= max_tokens:
            tokenized_sent = tokenized_sent[:max_tokens-word_counter]

        word_counter += len(tokenized_sent)
        cleaned_sentence = ' '.join(tokenized_sent)

        # if it is not empty
        if len(cleaned_sentence) > 0:
            # add Start of Sentence and End of Sentence special tokens
            cleaned_sentence_special_chars = '<SOS> ' + cleaned_sentence + ' <EOS>\n'

            # append the cleaned sentences to the cleaned collection
            cleaned_sentences.append(cleaned_sentence_special_chars)

            # only for training set
            if corpus_path:
                with open(corpus_path, 'a+', encoding="utf8") as corpus_file:
                    # update the corpus file, if this file is from
                    corpus_file.write(cleaned_sentence_special_chars)

    return  cleaned_sentences
