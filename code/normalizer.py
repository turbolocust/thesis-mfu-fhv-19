"""
author: Matthias Fussenegger
"""
from typing import List, Tuple
from random import seed
from random import shuffle
from random import randint
from math import ceil
import sys
import os
import re

IS_SHUFFLE_DATA = True  # true to shuffle data before cross folds
IS_SHUFFLE_FOLDS = True  # true to shuffle folds before processing them

SRC_SUFFIX = ".src"
TGT_SUFFIX = ".tgt"
DOC_SUFFIX = ".docid"
BBX_SUFFIX = ".bbox"


class Fold:

    def __init__(self, train_set: List,
                 dev_set: List,
                 test_set: List) -> None:
        super().__init__()
        assert len(train_set) != 0
        assert len(test_set) != 0
        self.__train_set = None
        self.__dev_set = None
        self.__test_set = None

        # process train set
        if isinstance(train_set[0], Document):
            self.__train_set = []
            for doc in train_set:
                self.__train_set.extend(doc.get_lines())
        else:
            self.__train_set = list(train_set)

        # process test set
        if isinstance(test_set[0], Document):
            self.__test_set = []
            for doc in test_set:
                self.__test_set.extend(doc.get_lines())
        else:
            self.__test_set = list(test_set)

        # process dev set
        if dev_set is not None \
                and len(dev_set) != 0:
            if isinstance(dev_set[0], Document):
                self.__dev_set = []
                for doc in dev_set:
                    self.__dev_set.extend(doc.get_lines())
            else:
                self.__dev_set = list(dev_set)

    def get_train_set(self) -> List[str]:
        return list(self.__train_set)

    def get_dev_set(self) -> List[str]:
        return list(self.__dev_set)

    def get_test_set(self) -> List[str]:
        return list(self.__test_set)


# noinspection SpellCheckingInspection
class Document:

    def __init__(self, ngrams: List[str]) -> None:
        super().__init__()
        self.__ngrams = list(ngrams)

    def get_lines(self) -> List[str]:
        return self.__ngrams

    @staticmethod
    def extract_documents(lines: List[str]) -> List:
        re_doc = re.compile(r"\w{40}")  # SHA-1 (40 chars)

        docs = []
        doc_lines = []
        cur_docid = None

        for line in lines:
            match = re_doc.search(line)
            assert match is not None
            docid = match.group(0)
            if cur_docid is None:  # is first
                cur_docid = docid
                doc_lines.append(line)
            elif docid == cur_docid:
                doc_lines.append(line)
            else:  # new document
                doc = Document(doc_lines)
                docs.append(doc)
                doc_lines.clear()
                doc_lines.append(line)
                cur_docid = docid

        # consider last document
        doc = Document(doc_lines)
        docs.append(doc)

        return docs


class Directory:

    def __init__(self, path: str) -> None:
        super().__init__()
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.__path = path

    def mkdir(self, dir_name: str) -> str:
        path = os.path.join(self.__path, dir_name)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass  # ignore if already exists

        return path


# noinspection SpellCheckingInspection
class OutputGenerator:
    __re_src = re.compile(r"\s\t.+")
    __re_tgt0 = re.compile(r"\t(\w| |_)+\t")
    __re_doc = re.compile(r"\w{40}")  # SHA-1 (40 chars)
    __re_bbox = re.compile(r".+\w{40}\t")

    def __init__(self) -> None:
        super().__init__()

    def __prepare_src_set(self, data: List[str]) -> List[str]:
        src_set = []
        for line in data:
            if line.isspace():
                continue
            repl = self.__re_src.sub("", line)
            src_set.append(repl)

        return src_set

    def __prepare_tgt_set(self, data: List[str]) -> List[str]:
        tgt_set = []
        for line in data:
            if line.isspace():
                continue
            repl = self.__re_tgt0.search(line).group(0)
            repl = repl.replace("\t", "")
            # translate labels to more meaningful words
            # labels originated from SAP system (table)
            repl = repl.replace("UNKNOWN", "unbekannt")
            repl = repl.replace("XBLNR", "rechnungsnummer")
            repl = repl.replace("EBELN", "bestellnummer")
            repl = repl.replace("REDAT", "rechnungsdatum")
            repl = repl.replace("WRBTR", "gesamtbetrag")
            repl = repl.replace("WMWST", "steuerbetrag")
            repl = repl.replace("VAT_NUMBER", "uid-nummer")
            tgt_set.append(repl + "\n")

        if tgt_set[-1].isspace():
            del tgt_set[-1]

        return tgt_set

    def __prepare_doc_set(self, data: List[str]) -> List[str]:
        doc_set = []
        for line in data:
            if line.isspace():
                continue
            docid = self.__re_doc.search(line).group(0)
            doc_set.append(docid + "\n")

        if doc_set[-1].isspace():
            del doc_set[-1]

        return doc_set

    def __prepare_bbx_set(self, data: List[str]) -> List[str]:
        bbox_set = []
        for line in data:
            if line.isspace():
                continue
            bbox = self.__re_bbox.sub("", line)
            bbox_set.append(bbox)

        return bbox_set

    def generate(self, fold: Fold) -> Tuple[List[str], str]:
        # process training set
        src_set = self.__prepare_src_set(fold.get_train_set())
        tgt_set = self.__prepare_tgt_set(fold.get_train_set())
        doc_set = self.__prepare_doc_set(fold.get_train_set())
        bbx_set = self.__prepare_bbx_set(fold.get_train_set())
        yield src_set, "train" + SRC_SUFFIX
        yield tgt_set, "train" + TGT_SUFFIX
        yield doc_set, "train" + DOC_SUFFIX
        yield bbx_set, "train" + BBX_SUFFIX
        # process test set
        src_set = self.__prepare_src_set(fold.get_test_set())
        tgt_set = self.__prepare_tgt_set(fold.get_test_set())
        doc_set = self.__prepare_doc_set(fold.get_test_set())
        bbx_set = self.__prepare_bbx_set(fold.get_test_set())
        yield src_set, "test" + SRC_SUFFIX
        yield tgt_set, "test" + TGT_SUFFIX
        yield doc_set, "test" + DOC_SUFFIX
        yield bbx_set, "test" + BBX_SUFFIX
        # process dev set
        src_set = self.__prepare_src_set(fold.get_dev_set())
        tgt_set = self.__prepare_tgt_set(fold.get_dev_set())
        doc_set = self.__prepare_doc_set(fold.get_dev_set())
        bbx_set = self.__prepare_bbx_set(fold.get_dev_set())
        yield src_set, "dev" + SRC_SUFFIX
        yield tgt_set, "dev" + TGT_SUFFIX
        yield doc_set, "dev" + DOC_SUFFIX
        yield bbx_set, "dev" + BBX_SUFFIX


# noinspection SpellCheckingInspection
class FoldUtils:
    @staticmethod
    def __chunks_gen(dataset: List, steps: int) -> List:
        for i in range(0, len(dataset), steps):
            yield dataset[i:i + steps]

    @staticmethod
    def kfold_split(dataset: List, num_folds: int,
                    is_shuffle: bool = False) -> List[Fold]:

        assert num_folds > 2

        chunks = []
        folds = []  # to be returned

        fold_size = ceil(len(dataset) / num_folds)
        for chunk in FoldUtils. \
                __chunks_gen(dataset, fold_size):
            print("Length of chunk: %s" % str(len(chunk)))
            chunks.append(chunk)

        if is_shuffle:
            seed()  # seed is current time
            shuffle(chunks)

        for tst_idx in range(0, len(chunks)):
            tst = chunks[tst_idx]  # select as test set
            dev_idx = randint(0, len(chunks) - 1)
            while dev_idx == tst_idx:
                # randomly select dev set
                dev_idx = randint(0, len(chunks) - 1)
            dev = chunks[dev_idx]  # select as dev set
            train = []  # allocate new memory
            # build train set from remaining splits
            for i in range(0, len(chunks)):
                if i != tst_idx and i != dev_idx:
                    train.extend(chunks[i])

            fold = Fold(train_set=train,
                        dev_set=dev,
                        test_set=tst)
            folds.append(fold)

        return folds


def read_file(filename: str, encoding: str = "utf-8") -> List[str]:
    with open(filename, "r", encoding=encoding) as file:
        lines = file.readlines()

    return lines


def write_file(filename: str, lines: List[str]) -> None:
    with open(filename, "w", encoding="utf-8") as out_file:
        for line in lines:
            out_file.write(line)


def check_shuffle(data: List[str]) -> None:
    if IS_SHUFFLE_DATA:
        seed()  # seed is time
        shuffle(data)  # in-place


# noinspection SpellCheckingInspection
def main():
    ngrams_file = sys.argv[1]
    target_dir = sys.argv[2]
    num_folds = int(sys.argv[3])
    split_by_doc = sys.argv[4].lower()
    split_by_doc = True if split_by_doc == "true" else False

    directory = Directory(target_dir)
    lines = read_file(ngrams_file)

    if bool(split_by_doc):
        docs = Document.extract_documents(lines)
        check_shuffle(docs)  # happens in-place
        folds = FoldUtils.kfold_split(
            docs, num_folds, IS_SHUFFLE_FOLDS)
    else:  # split lines directly
        check_shuffle(lines)  # happens in-place
        folds = FoldUtils.kfold_split(
            lines, num_folds, IS_SHUFFLE_FOLDS)

    fold_suffix = 0
    output_gen = OutputGenerator()

    for fold in folds:
        fold_suffix += 1
        # create folder for each fold
        fold_dir = "FOLD" + str(fold_suffix)
        path = directory.mkdir(fold_dir)
        print("Fold directory created: %s" % path)
        # prepare folded sets and write to directory
        for out_set, name in output_gen.generate(fold):
            filename = os.path.join(path, name)
            write_file(filename, out_set)
            print("Set written to: %s" % filename)


if __name__ == "__main__":
    main()
