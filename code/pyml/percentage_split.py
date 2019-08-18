"""
author: Matthias Fussenegger
"""
import sys
from random import shuffle

IS_SHUFFLE = False


def main():
    split = sys.argv[1]
    fname = sys.argv[2]
    tname = sys.argv[3]
    with open(fname, "r", encoding="utf-8") as file:
        lines = file.readlines()
    if IS_SHUFFLE:
        shuffle(lines)
    r1 = int(len(lines) * (int(split) / 100))
    r2 = int(len(lines) * ((100 - int(split)) / 100))
    s1 = lines[:r1]
    s2 = lines[r1 + 1:]

    with open(tname + str(r1 + 1), "w", encoding="utf-8") as file:
        for line in s1:
            if line[-1] != "\n":
                line = line + "\n"
            file.write(line)
    with open(tname + str(r2), "w", encoding="utf-8") as file:
        for line in s2:
            if line[-1] != "\n":
                line = line + "\n"
            file.write(line)


if __name__ == "__main__":
    main()
