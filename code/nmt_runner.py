"""
author: Matthias Fussenegger
"""
from typing import List, Tuple
import sys
import os

COMPARE_ONLY = False  # skips training

# output directory of NMT model
OUT_DIR_NAME = "MODEL"
# prefixes for data sets
TRAIN_PREFIX = "train"
TEST_PREFIX = "test"
DEV_PREFIX = "dev"
# suffixes for each data set
SRC_SUFFIX = "src"
TGT_SUFFIX = "tgt"

# launch command for NMT
LAUNCH_COMM = "--src={0}" + " " + \
              "--tgt={1}" + " " + \
              "--vocab_prefix=\"{2}\"" + " " + \
              "--train_prefix=\"{3}\"" + " " + \
              "--embed_prefix=\"{4}\"" + " " + \
              "--dev_prefix=\"{5}\"" + " " + \
              "--test_prefix=\"{6}\"" + " " + \
              "--out_dir=\"{7}\"" + " " + \
              "--num_train_steps=12000" + " " + \
              "--num_enc_emb_partitions=1" + " " + \
              "--num_dec_emb_partitions=1" + " " + \
              "--steps_per_stats=100" + " " + \
              "--num_layers=2" + " " + \
              "--num_units=128" + " " + \
              "--dropout=0.2" + " " + \
              "--metrics=accuracy"


class NmtRunner:

    # noinspection PyCompatibility
    def __init__(self, vocab: str, embed: str) -> None:
        super().__init__()
        self.__vocab = ""
        self.__embed = ""

        ext_idx = vocab.rfind(".")
        if ext_idx != -1:
            self.__vocab = vocab[0:ext_idx]
        else:
            self.__vocab = vocab

        ext_idx = embed.rfind(".")
        if ext_idx != -1:
            self.__embed = embed[0:ext_idx]
        else:
            self.__embed = embed

    def run(self, root: str, path: str) -> bool:
        if COMPARE_ONLY:
            return True
        # build launch command
        params = LAUNCH_COMM.replace("{0}", SRC_SUFFIX)
        params = params.replace("{1}", TGT_SUFFIX)
        params = params.replace("{2}", self.__vocab)
        params = params.replace("{3}", os.path.join(path, TRAIN_PREFIX))
        if len(self.__embed) != 0:  # pre-trained embeddings are specified
            params = params.replace("{4}", self.__embed)
        else:  # remove parameters for pre-trained embeddings
            params = params.replace("--embed_prefix=\"{4}\"", "")
            params = params.replace("--num_enc_emb_partitions=1", "")
            params = params.replace("--num_dec_emb_partitions=1", "")
        params = params.replace("{5}", os.path.join(path, DEV_PREFIX))
        params = params.replace("{6}", os.path.join(path, TEST_PREFIX))
        params = params.replace("{7}", os.path.join(path, OUT_DIR_NAME))
        launch_comm = "python -m nmt.nmt " + params
        rc = os.system("cd \"" + root + "\"")  # switch to NMT directory
        assert rc == 0  # must succeed
        # start training (busy waiting)
        return os.system(launch_comm) == 0


def get_dirs(path: str) -> List[str]:
    # get all directories from path
    dirs = [file for file in os.listdir(path)
            if os.path.isdir(os.path.join(path, file))]
    return dirs


def compare_output(script: str, src: str, tgt: str, out: str) -> bool:
    if os.path.exists(out):
        print("File will be replaced: %s" % out)
    # run file comparison and save output to file
    launch_comm = "python \"" + script + "\" " + \
                  "\"" + src + "\"" + " " + "\"" + tgt + "\"" + \
                  " > " + "\"" + out + "\""
    return os.system(launch_comm) == 0


def main():
    nmt_root = sys.argv[1]  # root directory of NMT project
    working_dir = sys.argv[2]  # directory where folds are located
    fold_dir_prefix = sys.argv[3]  # prefix of folder name of fold
    vocab_name = sys.argv[4]  # path to vocabulary (full name, prefix only)
    embed_name = sys.argv[5]  # path to pre-trained embeddings (full name, prefix only)
    comp_script = sys.argv[6]  # path to script for file comparison

    nmt_runner = NmtRunner(vocab_name, embed_name)

    for d in get_dirs(working_dir):
        dir_name = d  # is just the name, no path included
        if dir_name.startswith(fold_dir_prefix):
            dir_name = os.path.join(working_dir, dir_name)
            print("Detected Fold in: %s" % dir_name)
            if not nmt_runner.run(nmt_root, dir_name):
                print("Error running NMT with Fold: %s" % dir_name)
            else:  # success
                # compare test output and write to file
                src_tst = os.path.join(dir_name, TEST_PREFIX + "." + TGT_SUFFIX)
                tgt_tst = os.path.join(dir_name, OUT_DIR_NAME, "output_test")
                out_tst = os.path.join(dir_name, OUT_DIR_NAME, "comp_test.txt")
                compare_output(comp_script, src_tst, tgt_tst, out_tst)
                # compare dev output and write to file
                src_dev = os.path.join(dir_name, DEV_PREFIX + "." + TGT_SUFFIX)
                tgt_dev = os.path.join(dir_name, OUT_DIR_NAME, "output_dev")
                out_dev = os.path.join(dir_name, OUT_DIR_NAME, "comp_dev.txt")
                compare_output(comp_script, src_dev, tgt_dev, out_dev)


if __name__ == "__main__":
    main()
