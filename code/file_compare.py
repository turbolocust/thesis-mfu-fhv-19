"""
author: Matthias Fussenegger
"""
import sys

UNK_LABEL = "unbekannt"

f1 = sys.argv[1]  # actual labels
f2 = sys.argv[2]  # predicted labels

with open(f1, "r", encoding="utf-8") as file:
    actual_l = file.readlines()

with open(f2, "r", encoding="utf-8") as file:
    predicted_l = file.readlines()

false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0

diff_count = 0
count_unk = 0
total_tokens = 0
diff_lines = []

for i, line in enumerate(actual_l):
    if line.isspace():
        continue  # skip empty lines

    actual = str(line)  # copy
    actual = actual.replace("\n", "")
    actual = actual.replace("\r", "")

    prediction = str(predicted_l[i])
    prediction = prediction.replace("\n", "")
    prediction = prediction.replace("\r", "")

    act_labels = actual.split()  # split on whitespace
    prd_labels = prediction.split()  # split on whitespace

    for ii, act_label in enumerate(act_labels):
        total_tokens += 1
        if act_label.startswith(UNK_LABEL):
            count_unk += 1
        prd_label = prd_labels[ii]
        # check equality of lines
        if act_label != prd_label:
            diff_count += 1
            out_line = act_label + " / " + prd_label + \
                       "\t" + actual + " / " + prediction
            diff_lines.append(out_line)
            if prd_label.startswith(UNK_LABEL):
                false_negatives += 1
            else:
                false_positives += 1
        elif act_label.startswith(UNK_LABEL) and \
                prd_label.startswith(UNK_LABEL):
            true_negatives += 1  # correct "unknown"
        else:  # score without "unknown" only label(s)
            true_positives += 1

count_not_unk = total_tokens - count_unk

print("\n")
print("Total tokens: %s" % str(total_tokens))
print("Different tokens: %s" % str(diff_count))
print("\nLabels (excl. '%s') present: %s " % (UNK_LABEL, str(count_not_unk)))

acc_nn = 0.0
acc_eu = 0.0

if total_tokens != 0:
    acc_nn = (total_tokens - diff_count) / float(total_tokens)

if count_not_unk != 0:
    acc_eu = true_positives / float(count_not_unk)

print("\nAccuracy (Total): %s" % str(acc_nn))
print("Accuracy (excl. Unknown): %s" % str(acc_eu))

print("\nMatches (excluding '%s'): %s" % (UNK_LABEL, str(true_positives)))
print("Matches (including '%s'): %s" % (UNK_LABEL, str(true_negatives + true_positives)))

# precision and recall
precision = true_positives / (float(true_positives) + false_positives)
recall = true_positives / (float(true_positives) + false_negatives)
if precision == 0 and recall == 0:
    precision = recall = -1
f1_score = 2 * ((precision * recall) / (precision + recall))

print("\nF1-Score: %s" % str(f1_score))

# print all lines that are different
print("\nDifferent:")
print("================")
for line in diff_lines:
    print(line)
