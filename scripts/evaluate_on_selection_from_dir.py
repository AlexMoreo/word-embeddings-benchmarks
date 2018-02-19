from optparse import OptionParser
import logging
import os, sys
from web.embedding import Embedding
from web.embeddings import load_embedding
from web.datasets.utils import _get_dataset_dir
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from web.evaluate import *

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

parser = OptionParser()
parser.add_option("-d", "--dir", dest="directory",
                  help="Path to the directory containing the embedding files.",
                  default=None)

parser.add_option("-o", "--output", dest="output",
                  help="Path where to save results.",
                  default=None)

parser.add_option("-c", "--clean_words", dest="clean_words",
                  help="Clean_words argument passed to load_embedding function. If set to True will remove"
                       "most of the non-alphanumeric characters, which should speed up evaluation.",
                  default=False)

def printflush(msg):
    print(msg,end='')
    sys.stdout.flush()

def get_metadata_from_name(name):
    name_parts = name.split('_')
    method,words,ctx,dim = name_parts[:4]
    ctx = ctx.replace('ctx','')
    dim = dim.replace('dim', '')
    k,config='',''
    if len(name_parts)>4:
        k = name_parts[4].replace('k','')
        config = '_'.join(name_parts[5:])
    metadata = {'method':method, 'words':words, 'ctx':ctx, 'dim':dim, 'k':k, 'config':config}
    return pd.DataFrame([metadata])

if __name__ == "__main__":
    (options, args) = parser.parse_args()

    out_fname = options.output if options.output else "results_selection.csv"
    results, computed = None, None
    if os.path.exists(out_fname):
        results = pd.read_csv(out_fname)
        computed = np.unique(results.embeddings)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Load embeddings
    dname = options.directory
    assert dname is not None, 'directory not specified. Abort.'

    wordfiles = [f for f in listdir(dname) if isfile(join(dname, f))]
    for wordfile in wordfiles:
        format = 'word2vec_bin'
        embeddings_name = wordfile
        if wordfile.endswith('.dict.pickle'):
            # os.path.basename(fname)
            embeddings_name = wordfile.replace('.dict.pickle', '')
            format = 'dict'

        printflush('Processing '+wordfile+ ' ')
        if computed is not None and embeddings_name in computed:
            print('[already computed]')
        else:
            printflush('[loading...')
            w = load_embedding(join(dname, wordfile), format=format, normalize=True, lower=True,
                               clean_words=options.clean_words, name=embeddings_name)
            printflush('OK][evaluating...')
            embedding_results = evaluate_on_selection(w)
            metadata = get_metadata_from_name(embeddings_name)
            embedding_results = embedding_results.join(metadata)
            if results is not None:
                results = results.append(embedding_results)
            else:
                results = embedding_results
            print('OK]')
            results.to_csv(out_fname, index_label='embeddings', index=False) #just in case

    print("Done")

    print(results)
