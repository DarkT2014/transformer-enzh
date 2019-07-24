#-*-coding:utf-8-*-
from __future__ import division
import sys
import os
import io
import time

_package_path = "/".join(os.path.abspath(os.path.dirname(__file__)).split("/")[:-1])
sys.path.append(_package_path)

import tensorflow as tf
import jieba


MIN_COUNT = 5

_TRAIN_DATA = {
    'inputs': 'train.en',
    'targets': 'train.zh'
}


_DEV_DATA = {
    'inputs': 'dev.en',
    'targets': 'dev.zh'
}




def iterator_file(file_path):
    with io.open(file_path, encoding='utf-8') as inf:
        for i, line in enumerate(inf):
            yield i,line


_PREFIX = "translate"

def shard_filename(path, tag, shard_num, total_shards):
    """Create filename for data shard."""
    return os.path.join(
        path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))

def dict_to_example(dictionary):
    """Converts a dictionary of string->int to a tf.Example."""
    features = {}
    for k, v in dictionary.items():
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))

# from transformer import textencoder

from transformer.utils import tokenizer

def create_tf_record(source_files, vocab_files, out_dir, mode, total_shards):
    input_encoder = tokenizer.Subtokenizer(vocab_files[0])
    target_encoder = tokenizer.Subtokenizer(vocab_files[1])
    shard_files = [shard_filename(out_dir, mode, n + 1, total_shards) for n in range(total_shards)]
    writers = [tf.python_io.TFRecordWriter(fname) for fname in shard_files]

    input_file = source_files[0]
    target_file = source_files[1]
    counter = 0
    shard = 0
    for input_line, target_line in zip(iterator_file(input_file), iterator_file(target_file)):
        counter += 1

        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("\tSaving case %d." % counter)

        example_dict = {
            'inputs':  input_encoder.encode(input_line, True),
            'targets': target_encoder.encode(target_line, True)
        }

        example = dict_to_example(example_dict)
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % total_shards


def write_file(outf, content_list):
    for line in content_list:
        outf.write(line + "\n")



if __name__ == '__main__':

    # source_dir = "./test_data"
    source_dir = "/tmp/t2t_datagen"


    ##### create vocab
    zh_vocab = os.path.join(source_dir, "zh_sub_word.vocab")
    en_vocab = os.path.join(source_dir, "en_sub_word.vocab")
    print("create vocab ...")


    en_source_file = os.path.join(source_dir, _TRAIN_DATA['inputs'])
    zh_source_file = os.path.join(source_dir, _TRAIN_DATA['targets'])


    inputs_tokenizer = tokenizer.Subtokenizer.init_from_files(
        en_vocab, [en_source_file],  2**15, 20,
        min_count=None, file_byte_limit=1e8)
    print("en tokenize done.")
    # targets_tokenizer = tokenizer.Subtokenizer.init_from_files(
        # zh_vocab, [zh_source_file],  2**15, 20,
        # min_count=None, file_byte_limit=1e8)
    zh_subtoken_list = []
    totalLineNum = os.popen('wc -l ' + zh_source_file).read().split()[0]
    global start_time
    start_time = time.time()
    for i, line in iterator_file(zh_source_file):
        line = line.replace("\r","").replace("\n","")
        percent = i/int(totalLineNum) * 100
        duration = time.time() - start_time
        sys.stdout.write("\r%.2f%% cutting %dth line of %d, %d sec passed. "% (percent, i, int(totalLineNum), duration))
        # print("\rcutting " + str(i) + "th line of :" + line, end='', flush=True)
        sys.stdout.flush()
        zh_subtoken_list.extend(jieba.lcut(line))
    print("cut list done..")
    
    print("tokenizing zh vocab..")
    targets_tokenizer = tokenizer.Subtokenizer(zh_vocab, zh_subtoken_list)
    print("tokenizing zh vocab done.")


    # data_dir = './train_data'
    data_dir = '/tmp/t2t_datagen/'

    print("create train_tfrecord")

    create_tf_record([en_source_file, zh_source_file], [en_vocab, zh_vocab], data_dir, 'train', 10 )

    create_tf_record([os.path.join(source_dir, _DEV_DATA['inputs']), os.path.join(source_dir, _DEV_DATA['targets'])],
                     [en_vocab, zh_vocab],
                     data_dir,
                     'dev',
                     1)
