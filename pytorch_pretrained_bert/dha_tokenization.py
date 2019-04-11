# -*- coding: utf-8 -*-


"""
DHA example code for Python Wrapper
"""

import dha
import json
import collections
import six
import os
import multiprocessing
import itertools

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

DHAToken = collections.namedtuple('DHAToken', ['cpos', 'clen', 'str', 'tag'])

class DHAAnalyzer(object):
    def __init__(self):
        self.coll_name = 'default'
        self.anal_name = 'hanl'
        self.options = ['sample|verb_root|ncp_root|-ncp_root|level_9|-d_tag|m_tag|pos_longest|-text|char_pos']
        # self.options = ['indexing']
        self.dha_obj = dha.DHA(None)
        res_dir = os.environ['DHA_RES_DIR']
        self.dha_obj.initialize(res_dir, None, self.coll_name)    # 사전경로와 컬렉션명으로 초기화를 합니다.

    @staticmethod
    def tokens_str(tokens):
        return [t.str for t in tokens]

    def analyze(self, text, only_str):
        results = self.dha_obj.analyze(text, self.anal_name, self.options)
        tokens = [DHAToken(**t) for t in json.loads(results[0])]

        if only_str:
            return self.tokens_str(tokens)
        else:
            return tokens

def analyze_fn(args):
    text, only_str = args
    return analyzer.analyze(text, only_str)


class DHATokenizer(object):
    def __init__(self, vocab_file, max_workers = None, no_analyzer=False):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.UNK_ID = self.vocab.get('[UNK]')

        self.pool = None
        if max_workers:
            self.create_dha_pool(max_workers)
        elif not no_analyzer:
            self.dha_analyzer = DHAAnalyzer()

    def create_dha_pool(self, max_workers):
        def initializer():
            global analyzer
            analyzer = DHAAnalyzer()
        self.pool = multiprocessing.Pool(max_workers, initializer=initializer)

    def tokenize(self, text, only_str = True):
        input_is_list = isinstance(text, list)
        if input_is_list:
            text_list = text
        else:
            text_list = [text]

        if self.pool:
            result = self.pool.map(analyze_fn, itertools.product(text_list, [only_str]))
        else:
            result = []
            for text in text_list:
                result.append(self.dha_analyzer.analyze(text, only_str))

        if input_is_list:
            return result
        else:
            return result[0]


    def convert_tokens_to_ids(self, tokens):
        output = []
        for t in tokens:
            vocab_id = self.vocab.get(t)
            if vocab_id is None:
                vocab_id = self.UNK_ID
            output.append(vocab_id)
        return output

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for vocab_id in ids:
            output.append(self.inv_vocab[vocab_id])
        return output


def main():
    vocab_file = 'samples/hangul_vocab.txt'
    tokenizer = DHATokenizer(vocab_file, 4)
    tokens = tokenizer.tokenize("철수가 밥을 먹고 학교에 갔다.", only_str=False)
    print(json.dumps(tokens, ensure_ascii=False))
    tokens = tokenizer.tokenize("철수가 밥을 먹고 학교에 갔다.")
    print(' '.join(tokens))
    print(tokenizer.convert_tokens_to_ids(tokens))

    input_text = ['철수가 밥먹었니?', '나는 오늘 윈드서핑 하러 갈거야']
    print(tokenizer.tokenize(input_text))


if __name__ == '__main__':
    main()
