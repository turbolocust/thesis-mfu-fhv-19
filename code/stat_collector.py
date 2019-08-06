"""
author: Matthias Fussenegger
"""
from typing import List, Tuple
import statistics as stat
import scipy.stats as st
import sys
import os
import re

# CONSTANTS
NORM_DISTRIBUTION = False
MODEL_FOLDER = "MODEL"
COMP_DEV_FILE = "comp_dev.txt"
COMP_TST_FILE = "comp_test.txt"
REGEX_NUMBER = re.compile(r"\d+")
REGEX_DECIMAL = re.compile(r"\d{0,3}\.\d+")


class ResultRow:

    def __init__(self, filename) -> None:
        super().__init__()
        self.__filename = filename
        self.f1_score = ""
        self.fold_num = ""
        self.ngram_size = ""
        self.total_lines = ""
        self.diff_lines = ""
        self.acc_total = ""
        self.acc_excl_unk = ""
        self.matches_excl_unk = ""
        self.matches_incl_unk = ""

    def get_filename(self) -> str:
        return self.__filename

    def to_txt(self, incl_f1: bool = True) -> str:
        txt = self.fold_num + "\t" + self.ngram_size + "\t" \
              + self.total_lines + "\t" + self.diff_lines + "\t" \
              + self.acc_total + "\t" + self.acc_excl_unk + "\t" \
              + self.matches_excl_unk + "\t" + self.matches_incl_unk
        if incl_f1:
            txt += "\t" + str(self.f1_score)
        return txt


# noinspection SpellCheckingInspection
class Measures:

    def __init__(self) -> None:
        super().__init__()
        self.acc_avg_total = 0.0
        self.acc_avg_excl_unk = 0.0
        self.stdev_total = 0.0
        self.stdev_excl_unk = 0.0
        self.var_total = 0.0
        self.var_excl_unk = 0.0
        self.conf_total = (0.0, 0.0)
        self.conf_excl_unk = (0.0, 0.0)

    @staticmethod
    def __conf_norm(alpha: float, data: List[float], mean: float) -> Tuple[float, float]:
        tt_alpha = (1 + alpha) / 2  # to get a two-tail result
        return st.norm.interval(tt_alpha, loc=mean, scale=st.sem(data))

    @staticmethod
    def __conf_t(alpha: float, data: List[float], mean: float) -> Tuple[float, float]:
        tt_alpha = (1 + alpha) / 2  # to get a two-tail result
        return st.t.interval(tt_alpha, len(data) - 1, loc=mean, scale=st.sem(data))

    def to_txt(self) -> str:
        return str(self.acc_avg_total) + "\t" + str(self.acc_avg_excl_unk) + "\t" \
               + str(self.stdev_total) + "\t" + str(self.stdev_excl_unk) + "\t" \
               + str(self.var_total) + "\t" + str(self.var_excl_unk) + "\t" \
               + str(self.conf_total[0]) + "\t" + str(self.conf_total[1]) + "\t" \
               + str(self.conf_excl_unk[0]) + "\t" + str(self.conf_excl_unk[1])

    @staticmethod
    def from_results(results: List[ResultRow], confidence: float):
        """
        Factory method which returns a new instance of this class.
        :param confidence: alpha for confidence interval.
        :param results: results which are used to calculate measures.
        :return: an instance of this class.
        """
        meas = Measures()  # to be returned
        # calculate average of total accuracy
        acc_total = [float(result.acc_total) for result in results]
        meas.acc_avg_total = sum(acc_total) / len(results)
        # calculate average of accuracy excluding UNK label
        acc_excl_unk = [float(result.acc_excl_unk) for result in results]
        meas.acc_avg_excl_unk = sum(acc_excl_unk) / len(results)
        # calculate standard deviation of average of total accuracy
        meas.stdev_total = stat.stdev([val * 100 for val in acc_total])
        # calculate standard deviation of average of accuracy excluding UNK label
        meas.stdev_excl_unk = stat.stdev([val * 100 for val in acc_excl_unk])
        # calculate variance of total accuracy
        meas.var_total = meas.stdev_total ** 2
        # calculate variance of accuracy excluding UNK label
        meas.var_excl_unk = meas.stdev_excl_unk ** 2
        # calculate confidence interval for both accuracies
        if NORM_DISTRIBUTION:
            meas.conf_total = Measures.__conf_norm(
                confidence, acc_total, mean=meas.acc_avg_total)
            meas.conf_excl_unk = Measures.__conf_norm(
                confidence, acc_excl_unk, mean=meas.acc_avg_excl_unk)
        else:  # student's t distribution
            meas.conf_total = Measures.__conf_t(
                confidence, acc_total, mean=meas.acc_avg_total)
            meas.conf_excl_unk = Measures.__conf_t(
                confidence, acc_excl_unk, mean=meas.acc_avg_excl_unk)

        return meas


def get_dirs(path: str, recurse: bool = False) -> List[Tuple[str, str]]:
    """
    Get all directories from path (optionally recursively).
    :param path: the path of which to get all directories.
    :param recurse: true to recursively search path.
    :return: list of tuples of (<root>, <dirname>).
    """
    dirs = []
    if recurse:
        for root, dirnames, _ in os.walk(path):
            dirs.extend([(root, dirname) for dirname in dirnames])
    else:  # only consider directories in specified path
        dirs = [(path, file) for file in os.listdir(path)
                if os.path.isdir(os.path.join(path, file))]

    return dirs


def read_file(filename: str, encoding: str = "utf-8") -> List[str]:
    with open(filename, "r", encoding=encoding) as file:
        lines = file.readlines()

    return lines


def write_file(filename: str, results: List[ResultRow], measures: Measures = None) -> None:
    with open(filename, "w", encoding="utf-8") as out_file:
        for result in results:
            txt_line = result.get_filename() + "\t"  # prepend filename first
            if measures is not None:  # include measures if provided
                txt_line += result.to_txt(False) + "\t" + \
                            measures.to_txt() + "\t" + \
                            str(result.f1_score)
            else:  # ignore measures and just print results
                txt_line += result.to_txt()

            out_file.write(txt_line + "\n")


# noinspection PyCompatibility
def eval_stat(filename: str, fold_num: str, ng_size: str) -> ResultRow:
    lines = read_file(filename)
    result: ResultRow = ResultRow(filename)
    result.fold_num = fold_num
    result.ngram_size = ng_size
    for line in lines:
        lower_l = line.lower().replace("\n", "").replace("\r", "")
        if lower_l.startswith("total tokens"):
            match = REGEX_NUMBER.search(lower_l)
            assert match is not None
            result.total_lines = match.group(0)
        elif lower_l.startswith("different tokens"):
            match = REGEX_NUMBER.search(lower_l)
            assert match is not None
            result.diff_lines = match.group(0)
        elif lower_l.startswith("accuracy (total)"):
            match = REGEX_DECIMAL.search(lower_l)
            assert match is not None
            result.acc_total = match.group(0)
        elif lower_l.startswith("accuracy (excl."):
            match = REGEX_DECIMAL.search(lower_l)
            assert match is not None
            result.acc_excl_unk = match.group(0)
        elif lower_l.startswith("matches (excluding"):
            match = REGEX_NUMBER.search(lower_l)
            assert match is not None
            result.matches_excl_unk = match.group(0)
        elif lower_l.startswith("matches (including"):
            match = REGEX_NUMBER.search(lower_l)
            assert match is not None
            result.matches_incl_unk = match.group(0)
        elif lower_l.startswith("f1-score"):
            match = REGEX_DECIMAL.search(lower_l)
            assert match is not None
            result.f1_score = match.group(0)
            break  # leave early

    return result


def comp_result(result: ResultRow) -> int:
    """
    Returns the key for comparison of the result object.
    :param result: the result object.
    :return: key, which is the fold number as integer.
    """
    return int(result.fold_num)


def main():
    working_dir = sys.argv[1]  # location of folds
    fold_prefix = sys.argv[2]  # prefix of folder name of fold
    output_file_prefix = sys.argv[3]  # prefix for results file
    ngram_size = sys.argv[4]  # is added to tabular output
    confidence = float(sys.argv[5])  # confidence for confidence interval

    recurse = False
    if len(sys.argv) == 7:
        recurse = True if sys.argv[6].lower() == "true" else False

    out_lines_dev = []
    out_lines_tst = []

    regex_txt = r"^" + fold_prefix + r"\d{0,5}$"
    regex_dir = re.compile(regex_txt)
    print("Evaluating: %s" % regex_txt)

    for root_dir, dir_name in get_dirs(working_dir, recurse=recurse):
        if regex_dir.match(dir_name) is not None:
            print("Fold detected: %s" % dir_name)
            match = REGEX_NUMBER.search(dir_name)
            if match is not None:
                fold_num = match.group(0)
            else:
                fold_num = ""  # leave empty if no number in folder name
            fullname = os.path.join(root_dir, dir_name, MODEL_FOLDER)
            print("Processing models in: %s" % fullname)
            comp_dev_name = os.path.join(fullname, COMP_DEV_FILE)
            comp_tst_name = os.path.join(fullname, COMP_TST_FILE)
            out_lines_dev.append(eval_stat(comp_dev_name, fold_num, ngram_size))
            out_lines_tst.append(eval_stat(comp_tst_name, fold_num, ngram_size))

    out_lines_dev.sort(key=comp_result)
    out_lines_tst.sort(key=comp_result)

    dev_file_name = output_file_prefix + "_dev.txt"
    tst_file_name = output_file_prefix + "_tst.txt"

    if len(out_lines_dev) != 0:
        meas_dev = Measures.from_results(out_lines_dev, confidence)
        write_file(os.path.join(working_dir, dev_file_name), out_lines_dev, meas_dev)
        print("DEV comparison file written to: %s" % dev_file_name)
    else:
        print("DEV comparison file is empty!")

    if len(out_lines_tst) != 0:
        meas_tst = Measures.from_results(out_lines_tst, confidence)
        write_file(os.path.join(working_dir, tst_file_name), out_lines_tst, meas_tst)
        print("TEST comparison file written to: %s" % tst_file_name)
    else:
        print("TEST comparison file is empty!")


if __name__ == "__main__":
    main()
