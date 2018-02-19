#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script calculates embedding results against all available fast running
 benchmarks in the repository and saves results as single row csv table.

 Usage: ./evaluate_on_all -f <path to file> -o <path to output file>

 NOTE:
 * script doesn't evaluate on WordRep (nor its subset) as it is non standard
 for now and long running (unless some nearest neighbor approximation is used).

 * script is using CosAdd for calculating analogy answer.

 * script is not reporting results per category (for instance semantic/syntactic) in analogy benchmarks.
 It is easy to change it by passing category parameter to evaluate_analogy function (see help).
"""
from optparse import OptionParser
import logging
import os, sys
from web.embedding import Embedding
from web.embeddings import fetch_GloVe, load_embedding
from web.datasets.utils import _get_dataset_dir
import pandas as pd
import numpy as np

from web.evaluate import evaluate_on_all, evaluate_on_selection

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-p", "--format", dest="format",
                  help="Format of the embedding, possible values are: word2vec, word2vec_bin, dict and glove.",
                  default=None)

parser.add_option("-o", "--output", dest="output",
                  help="Path where to save results.",
                  default=None)

parser.add_option("-c", "--clean_words", dest="clean_words",
                  help="Clean_words argument passed to load_embedding function. If set to True will remove"
                       "most of the non-alphanumeric characters, which should speed up evaluation.",
                  default=False)

if __name__ == "__main__":
    (options, args) = parser.parse_args()

    out_fname = options.output if options.output else "results_selection.csv"
    results = None
    if os.path.exists(out_fname):
        results = pd.read_csv(out_fname)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Load embeddings
    fname = options.filename
    if not fname:
        w = fetch_GloVe(corpus="wiki-6B", dim=50)
    else:
        if not os.path.isabs(fname):
            fname = os.path.join(_get_dataset_dir(), fname)

        embeddings_name = os.path.basename(fname).replace('.dict.pickle', '')
        format = options.format
        print('Loading file {}, embeddings name {}'.format(fname, embeddings_name))
        if not format:
            _, ext = os.path.splitext(fname)
            if ext == ".bin":
                format = "word2vec_bin"
            elif ext == ".txt":
                format = "word2vec"
            elif ext == ".pkl":
                format = "dict"

        assert format in ['word2vec_bin', 'word2vec', 'glove', 'bin', 'dict'], "Unrecognized format"

        load_kwargs = {}
        if format == "glove":
            load_kwargs['vocab_size'] = sum(1 for line in open(fname))
            load_kwargs['dim'] = len(next(open(fname)).split()) - 1

        w = load_embedding(fname, format=format, normalize=True, lower=True, clean_words=options.clean_words,
                           load_kwargs=load_kwargs, name=embeddings_name)

        print('Loaded {} embeddings'.format(len(w)))

    if results is not None and w.name in results['embeddings'].values:
        print("Embeddings {} already calculated in {}; nothing to do here".format(w.name, out_fname))
        print(results)
        sys.exit()

    embedding_results = evaluate_on_selection(w)

    if results is not None:
        results = results.append(embedding_results)
    else:
        results = embedding_results


    logger.info("Saving results...")
    print(results)
    results.to_csv(out_fname, index_label='embeddings', index=False)
