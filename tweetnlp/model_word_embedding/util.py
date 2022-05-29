""" Download files from web. If Gdrive downloading stacked, try to remove the gdown cache `rm -rf ~/.cache/gdown` """
import tarfile
import zipfile
import gzip
import requests
import os

import gdown
from gensim.models import KeyedVectors
from gensim.models import fasttext


urls = {
    'en': "https://drive.google.com/file/d/1E8T8PAPa-OafSRqB50awhF6LD9igP4So/view?usp=sharing",
    'ja': "https://drive.google.com/file/d/1A6SVYkqmtwCTYeFjTrBJuXdVb6UxWkfs/view?usp=sharing",
    'it': "https://drive.google.com/file/d/14zKLFAFLeL9P2ZG6JV6uBJvWAYxyLCUU/view?usp=sharing",
    'es': "https://drive.google.com/file/d/1J06BUDIjUQOtWkv7aM1AjS9e59Wf-OcG/view?usp=sharing",
    'du': "https://drive.google.com/file/d/12yb61Ar5KJMMgAf6T5yLB0UkS6Rnzn4U/view?usp=sharing",
    'fi': "https://drive.google.com/file/d/1hz4x0hEVPz4QyXN0drzhv-AmjlekNQSy/view?usp=sharing",
    'fas': "https://drive.google.com/file/d/1yP-JUSHriJCBGQiQVm46rW7W5cFv4DFh/view?usp=sharing"
}


def get_word_embedding_model(model_name: str = 'fasttext'):
    """ get word embedding model """
    os.makedirs('./cache', exist_ok=True)
    if model_name == 'w2v':
        path = './cache/GoogleNews-vectors-negative300.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url="https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download",
                cache_dir='./cache',
                gdrive_filename='GoogleNews-vectors-negative300.bin.gz'
            )
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'fasttext_cc':
        path = './cache/crawl-300d-2M-subword.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip',
                cache_dir='./cache')
        model = fasttext.load_facebook_model(path)
        # model = KeyedVectors.load_word2vec_format(path)
    elif model_name == 'fasttext':
        path = './cache/wiki-news-300d-1M.vec'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                cache_dir='./cache'
            )
        model = KeyedVectors.load_word2vec_format(path)
    elif model_name == 'glove':
        path = './cache/glove.840B.300d.gensim.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url='https://drive.google.com/u/0/uc?id=1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN&export=download',
                cache_dir='./cache',
                gdrive_filename='glove.840B.300d.gensim.bin.tar.gz'
            )
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'pair2vec':
        path = './cache/pair2vec.fasttext.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/pair2vec.fasttext.bin.tar.gz',
                cache_dir='./cache')
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        path = './cache/{}.bin'.format(model_name)
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(url='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/{}.bin.tar.gz'.format(model_name),
                 cache_dir='./cache')
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model


def wget(url, cache_dir: str, gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    path = _wget(url, cache_dir, gdrive_filename=gdrive_filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    # return path


def _wget(url: str, cache_dir, gdrive_filename: str = None):
    """ get data from web """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        return gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return '{}/{}'.format(cache_dir, filename)