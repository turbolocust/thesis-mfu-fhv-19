"""
author: Matthias Fussenegger
"""
import sys
import os

# Window Size, Embedding Dimension, Epochs
CONFIG = ["3,5,50000",
          "3,10,50000",
          "3,20,50000",
          "3,40,50000",
          "3,80,50000",
          "3,160,50000",
          "3,320,50000"]

FOLDER_PREFIX = "ID"
NAME_MODEL = "word2vec_model_gensim"
NAME_LOG_F = "gensim.log"


# noinspection SpellCheckingInspection
class GensimRunner:

    def __init__(self, script, corpus):
        self.__script = script
        self.__corpus = corpus
        self.__iteration = 0

    def run_gensim(self, conf: str, working_dir: str):
        values = conf.split(",")
        assert len(values) == 3
        win_size = int(values[0])
        embed_dim = int(values[1])
        max_epochs = int(values[2])

        suffix = self.__iteration
        self.__iteration += 1
        name = FOLDER_PREFIX + str(suffix)
        path = os.path.join(working_dir, name)

        try:
            os.mkdir(path)
        except FileExistsError:
            pass  # ignore

        model_file = os.path.join(path, NAME_MODEL)
        log_file = os.path.join(path, NAME_LOG_F)
        launch_comm = "python" + " \"" + self.__script + "\" " + \
                      "\"" + self.__corpus + "\" " + \
                      "\"" + model_file + "\" " + \
                      "\"" + log_file + "\" " + \
                      str(win_size) + " " + \
                      str(embed_dim) + " " + \
                      str(max_epochs)

        if os.system(launch_comm) == 0:
            print("Iteration %s successful." % str(suffix))
        else:
            print("Iteration %s errored." % str(suffix))


# noinspection SpellCheckingInspection
def main():
    gensim_script = sys.argv[1]
    corpus = sys.argv[2]

    try:
        working_dir = sys.argv[3]
    except IndexError:
        working_dir = ""  # local

    runner = GensimRunner(gensim_script, corpus)

    for conf in CONFIG:
        runner.run_gensim(conf, working_dir)


if __name__ == "__main__":
    main()
