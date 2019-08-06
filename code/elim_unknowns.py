"""
author: Matthias Fussenegger
"""
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]
pr = sys.argv[3]  # percentage of labels to eliminate

with open(f1, "r", encoding="utf-8") as file:
    l1 = file.readlines()

uc = 0

for line in l1:
    if "\tUNKNOWN\t" in line:
        uc += 1

us = uc * (float(pr) / 100)
c = 0
l2 = []

for line in l1:
    if "\tUNKNOWN\t" in line:
        if c >= us:  # don't skip anymore
            l2.append(line)
        else:  # skip
            c += 1
    else:
        l2.append(line)

with open(f2, "w", encoding="utf-8") as file:
    for line in l2:
        file.write(line)
