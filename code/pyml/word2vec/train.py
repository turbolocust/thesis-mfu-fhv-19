"""
author: Matthias Fussenegger

GENSIM library has a known bug with regards to loss calculation:
https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim
"""

from typing import List, AnyStr, Tuple
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import sys
import logging as log

# CONSTANTS
IS_LOAD_MODEL = True
FIND_SIMILAR_DN_POS = ["gesamtbetrag", "total"]
FIND_SIMILAR_DN_NEG = []
FIND_SIMILAR_DT_POS = ["datum", "rechnungsdatum"]
FIND_SIMILAR_DT_NEG = []

# GLOBALS
word_whitelist = {}


class EpochSaver(CallbackAny2Vec):
    def __init__(self, path):
        self.__epoch = 0
        self.__path = path
        self.__delay = 0

    def on_epoch_end(self, model):
        self.__delay += 1
        self.__epoch += 1
        if self.__delay == 100:  # save each 100 epoch
            model.save(self.__path)
            model.wv.save_word2vec_format(self.__path + "_word2vec.txt")
            self.__delay = 0


class EpochLogger(CallbackAny2Vec):
    PRINT_DELAY = 100

    def __init__(self):
        self.__epoch = 0
        self.__prev_loss = 0
        self.__print_delay = 0

    def on_epoch_end(self, model):
        self.__print_delay += 1
        self.__epoch += 1
        loss = model.get_latest_training_loss()
        cur_loss = loss - self.__prev_loss
        self.__prev_loss = loss
        loss_msg = "Loss is: %s" % str(cur_loss)
        epoch_msg = "Epoch: %s" % str(self.__epoch)
        if self.__print_delay == EpochLogger.PRINT_DELAY:
            print(loss_msg)
            print(epoch_msg)
            self.__print_delay = 0
        log.info(loss_msg)
        log.info(epoch_msg)
        # leave if loss is zero
        if cur_loss == 0:
            stop_msg = "Loss is zero."
            print(epoch_msg)
            print(stop_msg)
            log.info(stop_msg)
            sys.exit(stop_msg)


def read_file(filename: str, encoding: str = "utf-8") -> List[AnyStr]:
    with open(filename, "r", encoding=encoding) as file:
        lines = file.readlines()

    return lines


def parse_lines(lines: List[AnyStr]) -> str:
    text = ""
    for line in lines:
        if len(line) == 0 \
                or line.isspace() \
                or line.startswith("<<h>>"):
            continue
        text += line
    return text


def filter_word(word: str) -> bool:
    global word_whitelist
    if len(word_whitelist) != 0:
        if word not in word_whitelist:
            return True  # not on whitelist

    if word == '%':  # relevant for VAT
        return False

    if word == '.' or word == ':' or word == ',':
        return False

    if word == '/' or word == '\\':
        return False

    if word == "<s>" or word == "</s>":
        return True

    if len(word) < 2 or len(word) > 29:
        return True

    return False  # accept word


def generate_training_data(sentences: List[List[str]], win_size: int) \
        -> Tuple[List[list], List[List[str]]]:
    data = []
    sentences_f = []
    sentence_f = []
    # filter some words first
    for sentence in sentences:
        for word_index, word in enumerate(sentence):
            if not filter_word(word):
                sentence_f.append(word)
        if len(sentence_f) != 0:
            sentences_f.append(sentence_f.copy())
            sentence_f.clear()

    for sentence in sentences_f:
        for word_index, word in enumerate(sentence):
            # consider bounds for outer words
            for nb_word in sentence[
                           max(word_index - win_size, 0):
                           min(word_index + win_size, len(sentence)) + 1]:
                if nb_word != word:
                    data.append([word, nb_word])

    return data, sentences_f


def log_closest(w2v_model, pos_words, neg_words) -> None:
    loss = w2v_model.get_latest_training_loss()
    total_loss_msg = "Total loss is: %s" % str(loss)
    log.info(total_loss_msg)
    print(total_loss_msg)
    try:
        log.info("Positive words: %s" % str(pos_words))
        log.info("Negative words: %s" % str(neg_words))
        log.info(str(w2v_model.wv.most_similar(positive=pos_words, negative=neg_words)))
        log.info(str(w2v_model.wv.most_similar_cosmul(positive=pos_words, negative=neg_words)))
    except KeyError:
        pass


# noinspection SpellCheckingInspection
def main():
    global word_whitelist

    fname = sys.argv[1]  # first argument = corpus
    fname_out = sys.argv[2]  # second argument = output file
    fname_log = sys.argv[3]  # third argument = log file
    win_size = sys.argv[4]  # fourth argument = window wize
    embed_dim = sys.argv[5]  # fifth argument = embedding dimension
    num_epochs = sys.argv[6]  # sixth argument = number of epochs

    try:  # parse filename for whitelist
        fname_wl = sys.argv[7]  # seventh argument = whitelist
        lines_wl = read_file(fname_wl)
        lines_wl = [word.lower().strip('\n')
                    for word in lines_wl]
        word_whitelist = set(lines_wl)
    except IndexError:
        print("No whitelist specified")

    save_path_gs = fname_out
    save_path_logger = fname_log
    window_size = int(win_size)
    embedding_dim = int(embed_dim)
    epochs = int(num_epochs)

    # set up logger
    log.basicConfig(handlers=[log.FileHandler(save_path_logger, mode='w', encoding="utf-8")],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=log.INFO)

    print("Number of epochs: %s" % epochs)
    log.info("Number of epochs: %s" % epochs)

    lines = read_file(fname)
    corpus_raw = parse_lines(lines)
    corpus_raw = corpus_raw.lower()

    if IS_LOAD_MODEL:
        w2v_model = Word2Vec.load(save_path_gs)
        log_closest(w2v_model, FIND_SIMILAR_DN_POS, FIND_SIMILAR_DN_NEG)
        log_closest(w2v_model, FIND_SIMILAR_DT_POS, FIND_SIMILAR_DT_NEG)
    else:
        # raw sentences is a list of sentences
        corpus_raw = corpus_raw.replace("<s> ", "")
        corpus_raw = corpus_raw.replace(" </s>", "")
        raw_sentences = corpus_raw.split('\n')
        del corpus_raw  # reclaim memory
        sentences = []
        for sentence in raw_sentences:
            sentences.append(sentence.split())
        _, sentences = generate_training_data(sentences, window_size)
        # start training of word2vec model via gensim
        epoch_logger = EpochLogger()
        epoch_saver = EpochSaver(save_path_gs)
        # min_count = 2, data is being read via OCR, which produces errors
        w2v_model = Word2Vec(sentences, min_count=2, size=embedding_dim, window=window_size)
        try:
            w2v_model.train(sentences, total_examples=len(sentences), epochs=epochs,
                            compute_loss=True, callbacks=[epoch_saver, epoch_logger])
        except SystemExit:
            pass
        finally:  # save model
            w2v_model.save(save_path_gs)
            w2v_model.wv.save_word2vec_format(save_path_gs + ".src")
            log_closest(w2v_model, FIND_SIMILAR_DN_POS, FIND_SIMILAR_DN_NEG)
            log_closest(w2v_model, FIND_SIMILAR_DT_POS, FIND_SIMILAR_DT_NEG)


if __name__ == "__main__":
    main()
