"""
author: Matthias Fussenegger

Extracts tuple values and prints them from strings like:
[('r00', 0.3928552567958832), ('rech—dat', 0.34028685092926025)]
"""
from typing import List
import sys


def read_file(filename: str, encoding: str = "utf-8") -> List[str]:
    with open(filename, "r", encoding=encoding) as file:
        lines = file.readlines()

    return lines


def write_file(filename: str, lines: List[str]) -> None:
    with open(filename, "w", encoding="utf-8") as out_file:
        for line in lines:
            out_file.write(line + "\n")


# noinspection SpellCheckingInspection
def main():
    in_file: str = sys.argv[1]  # read input as file
    dest_path: str = ""  # destination path

    if len(sys.argv) == 3:
        dest_path = sys.argv[2]

    lines = []
    in_lines = read_file(in_file)

    for l in in_lines:
        n: str = l.strip()
        n = n[3:-2] if n[0] != '[' else n[2:-2]
        ss = n.split("), (")
        for s in ss:
            v = s.split(", ")
            t0, t1 = v[0].strip("'() "), \
                     v[1].strip("'() ")
            out_str: str = "{0}\t{1}".format(t0, t1)
            print(out_str)
            lines.append(out_str)

        lines.append("\n")

    if len(dest_path) != 0:
        write_file(dest_path, lines)


if __name__ == "__main__":
    main()
