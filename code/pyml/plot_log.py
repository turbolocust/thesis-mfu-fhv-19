"""
author: Matthias Fussenegger
"""
from typing import List
import os
import sys
import matplotlib.pyplot as plt


def read_logs_gen(log_dir: str, encoding: str) -> List[str]:
    """
    Reads log files in the specified directory.
    :param encoding: encoding of log files.
    :param log_dir: the directory which holds the log files.
    :return: an iterator over all log files.
    Each file is returned as a list of lines.
    """
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            with open(log_dir + "/" + filename,
                      encoding=encoding) as file:
                lines = file.readlines()
                yield lines


def main():
    log_dir = sys.argv[1]  # GENSIM log directory
    enc = sys.argv[2]  # encoding of log file
    enc = enc if len(enc) != 0 else "utf-8"

    if len(sys.argv) == 4:
        descend = sys.argv[3]  # only consider new values if lower
        descend = True if descend.lower() == "true" else False
    else:
        descend = False

    # collect all values
    for lines in read_logs_gen(log_dir, enc):
        # epoch = 0
        value = float("inf")  # infinity
        values = []
        # epochs = []
        for line in lines:
            if "Loss is:" in line:  # is line with loss value
                idx = line.index("Loss is:")
                if idx != -1:  # found
                    val = float(line[idx + 8:])
                    if descend and val < value:
                        value = val
                        values.append(val)
                    elif not descend:
                        values.append(val)
                    # epoch += 1
                    # epochs.append(epoch)

        values.sort(reverse=True)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(values)
        # noinspection PyBroadException
        try:
            plt.waitforbuttonpress()
            plt.show()
        except:
            pass  # ignore


if __name__ == "__main__":
    main()
