"""
author: Matthias Fussenegger
"""
from typing import List
import sys
import os


def read_file(filename: str, encoding: str = "utf-8") -> List[str]:
    with open(filename, "r", encoding=encoding) as file:
        lines = file.readlines()

    return lines


def write_file(filename: str, lines: List[str]) -> None:
    with open(filename, "w", encoding="utf-8") as out_file:
        for line in lines:
            out_file.write(line + "\n")


def main():
    sentences = sys.argv[1]  # source sentences (to be read)
    target_file = sys.argv[2]  # file to be created (vocabulary)

    vocab = set()
    vocab_lines = []

    for sentence in read_file(sentences):
        words = sentence.strip().split(" ")
        for word in words:
            if len(word) != 0:
                vocab.add(word)

    # build output list (to be stored away)
    for word in vocab:
        vocab_lines.append(word)

    write_file(target_file, vocab_lines)


if __name__ == "__main__":
    main()
