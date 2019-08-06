"""
author: Matthias Fussenegger
"""
from typing import List, Dict, Tuple
import sys
import os

# noinspection SpellCheckingInspection
"""
Examples:
unbekannt unbekannt rechnungsdatum / unbekannt unbekannt unbekannt
unbekannt rechnungsdatum unbekannt / unbekannt unbekannt unbekannt
rechnungsdatum unbekannt unbekannt / unbekannt unbekannt unbekannt
rechnungsnummer unbekannt rechnungsdatum / unbekannt unbekannt rechnungsdatum
"""

# noinspection SpellCheckingInspection
UNK_LABEL = "unbekannt"
LABELS = [UNK_LABEL, "rechnungsnummer", "rechnungsdatum",
          "gesamtbetrag", "steuerbetrag", "uid-nummer"]


class Counting:

    def __init__(self) -> None:
        super().__init__()
        self.__map = {}
        for label in LABELS:
            self.__map[label] = 0

    def count(self, label: str) -> int:
        count = self.__map[label] + 1
        self.__map[label] = count
        return count

    def get_values(self) -> Dict[str, int]:
        return self.__map.copy()

    def to_list(self, sep: str = "\t") -> List[str]:
        lines = []  # to be returned
        for key, value in self.__map.items():
            line = key + sep + str(value) + "\n"
            lines.append(line)
        return lines


# noinspection SpellCheckingInspection
class LinesProcessor:

    def __init__(self) -> None:
        super().__init__()
        self.__actuals = Counting()
        self.__predict = Counting()
        self.__predict_part = Counting()  # partially predicted
        self.__misclassified: Dict[str, Counting] = {}  # falsely predicted as
        for label in LABELS:
            self.__misclassified[label] = Counting()

    def process(self, lines: List[str]):
        for line in lines:
            if line.isspace():
                continue
            ngrams = line[line.index("\t"):]
            ngrams = ngrams.replace("\t", "")
            split = ngrams.split(" ")  # split at space
            ng_size = split.index("/")
            assert len(split) == (ng_size * 2) + 1
            for i in range(0, ng_size):
                ii = i + ng_size + 1
                f1 = split[i]
                f2 = split[ii]
                if ii == len(split) - 1:
                    f2 = f2.replace("\n", "").replace("\r", "")
                if f1 == f2:  # partially correct
                    self.__predict_part.count(f1)
                else:  # misclassified
                    self.__actuals.count(f1)
                    self.__predict.count(f2)
                    self.__misclassified[f1].count(f2)

    def get_result(self) -> List[str]:
        lines = self.__actuals.to_list()
        lines.insert(0, "!actuals\n")
        lines.append("\n!predicted\n")
        lines.extend(self.__predict.to_list())
        lines.append("\n!partially\n")
        lines.extend(self.__predict_part.to_list())
        lines.append("\n!misclassified\n")

        header = "\t"
        row = "{0}\t"
        idx_map: Dict[str, int] = {}
        # build header and row template (misclassified table)
        for i, label in enumerate(LABELS):
            header += label + "\t"
            idx_map[label] = i
            row += "{" + str(i + 1) + "}\t"

        lines.append(header + "\n")

        for label, counting in self.__misclassified.items():
            row_temp = str(row)  # copy
            repl = row_temp.replace("{0}", label)
            for col_label, count in counting.get_values().items():
                idx = idx_map[col_label] + 1  # one-based
                repl = repl.replace("{" + str(idx) + "}", str(count))
            # once done, append copied row
            lines.append(repl + "\n")

        return lines


def read_file(filename: str, encoding: str = "utf-8") -> List[str]:
    with open(filename, "r", encoding=encoding) as file:
        lines = file.readlines()

    return lines


def write_file(filename: str, lines: List[str]) -> None:
    with open(filename, "w", encoding="utf-8") as out_file:
        for line in lines:
            out_file.write(line)


def get_files(path: str, prefix: str, postfix: str) -> List[Tuple[str, str]]:
    return [(path, file) for file in os.listdir(path)
            if os.path.isfile(os.path.join(path, file))
            and file.startswith(prefix) and file.endswith(postfix)]


def store_away_result(filename: str, lines_proc: LinesProcessor) -> None:
    lines = lines_proc.get_result()
    write_file(filename, lines)


def main():
    src_dir = sys.argv[1]
    src_prefix = sys.argv[2]
    src_postfix = sys.argv[3]
    tgt_name = sys.argv[4]

    lines_proc = LinesProcessor()

    files = get_files(src_dir, src_prefix, src_postfix)
    for path, filename in files:
        lines = read_file(os.path.join(path, filename))
        lines_proc.process(lines)

    store_away_result(tgt_name, lines_proc)


if __name__ == "__main__":
    main()
