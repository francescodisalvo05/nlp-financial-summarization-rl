"""Evaluate models and decode output
- combined old Chen Rocks' eval_full_model.py and decode_full_model.py"""

import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat
from toolz.sandbox.core import unzip

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize
from data.data import CnnDmDataset
from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor

from metric import compute_rouge_n, compute_rouge_l_summ


DATA_DIR = None


class InferenceDataset(CnnDmDataset):
    def __init__(self, split, n_sentences):
        super().__init__(split, DATA_DIR)
        self._n_sentences = n_sentences

    def __getitem__(self, i):
        js_data = super().__getitem__(i)

        # take all the sentences
        if not self._n_sentences:
            art_sents = js_data['report']
            abs_sents = js_data['summary']

        # take the first "_n_sentences"
        # and filter out the ones extracted that are
        # greater than _n_sentences
        else:
            extracted = js_data['extracted']

            art_sents = js_data['report'][:self._n_sentences]
            abs_sents = []

            for idx,ex in enumerate(extracted):
                if ex < len(art_sents):
                    # we're not interested on the scores
                    abs_sents.append(js_data['summary'][idx])

            # edge case: no extracted sentences below the threshold
            # -> pick directly the summary instead of "rejecting" the data
            if not abs_sents:
                abs_sents = js_data['summary']

        return art_sents, abs_sents





def main(args, cuda):

    def coll(batch):
        art_batch, abs_batch = unzip(batch)
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        return art_sents, abs_sents

    dataset = InferenceDataset('test', args.n_sentences)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=coll
    )

    # setup abstractor
    with open(join(args.model_dir + '/meta.json')) as f:
        meta = json.loads(f.read())

    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        abstractor = identity
    else:
        abstractor = Abstractor(join(args.model_dir, 'abstractor'),
                                50, cuda)

    extractor = RLExtractor(args.model_dir, cuda=cuda)

    # setup output folder
    os.makedirs(join(args.output_path, 'txt'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True

    # save run logs
    with open(join(args.output_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # setuo scores
    rouge_2, rouge_l, bert_scores = [], [], []

    i = 0
    with torch.no_grad():
        for i_debug, (raw_article_batch,val_batch) in enumerate(loader):

            ext_arts = []
            ext_inds = []

            for raw_art_sents in raw_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]

            # evaluation
            decoded_outputs = abstractor(ext_arts)
            for (j, n), reference in zip(ext_inds, val_batch):

                for idx, sent in enumerate(reference):
                  # remove <sos>,<eos>
                  # otherwise they will count as match for rouge
                  reference[idx] = sent[1:-1]

                decoded_sentences = [' '.join(dec) for dec in decoded_outputs[j:j + n]]
                # evaluation
                rouge_2.append(compute_rouge_n(list(concat(decoded_outputs)), list(concat(reference)), n=2))
                rouge_l.append(compute_rouge_l_summ(decoded_outputs, reference))

                # to do
                # bert_scores.append()

                # postprocess output sentences
                cleaned_decoded_sentences = []
                for s in decoded_sentences:
                  s = s.replace("<sos>","").replace("<eos>","")[1:] # remove blank space
                  cleaned_decoded_sentences.append(s.capitalize())

                with open(join(args.output_path, f'txt/{i}.txt'), 'w', encoding="utf-8") as f:
                    f.write('\n'.join(cleaned_decoded_sentences))
                i += 1

    print(f'Rouge-2: {np.mean(np.asarray(rouge_2))}')
    print(f'Rouge-l: {np.mean(np.asarray(rouge_l))}')
    # print(f'BERTScore: {mean(bert_scores)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True, help='folder containing the extracted labels splitted in train and val')
    parser.add_argument('--model_dir', required=True, help='root of the models')
    parser.add_argument('--output_path', required=True, help='output path for the decoded reports')

    parser.add_argument('--batch_size', type=int, action='store', default=2,help='batch size of faster decoding')
    parser.add_argument('--n_sentences', type=int, action='store', default=None,help='maximum number of sentences used for decoding')

    args = parser.parse_args()

    cuda = torch.cuda.is_available()

    DATA_DIR = args.data_dir

    args = parser.parse_args()
    main(args,cuda)
