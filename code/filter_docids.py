"""
author: Matthias Fussenegger
"""
import sys
import re

RE_DOC = re.compile(r"\w{40}")  # SHA-1 (40 chars)

docids_file = sys.argv[1]
filter_file = sys.argv[2]
target_file = sys.argv[3]

with open(docids_file, "r", encoding="utf-8") as f:
    docs = f.readlines()

with open(filter_file, "r", encoding="utf-8") as f:
    filters = f.readlines()

filters = [f.replace("\r", "").replace("\n", "") for f in filters]
docid_filter = set(filters)
docs_target = []

for i, doc in enumerate(docs):
    if doc.isspace():
        continue
    docid = RE_DOC.search(doc).group(0)
    if docid in docid_filter:
        docs_target.append(doc)

with open(target_file, "w", encoding="utf-8") as f:
    f.writelines(docs_target)
