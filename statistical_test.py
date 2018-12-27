from scipy import stats
import matplotlib.pyplot as plt
import csv


def statistically_different(data1, data2, alpha=0.05):
    """
    alpha is the confidence level. It is initially 0.05 which means %95 confidence. 
    returns True if data1 and data2 statistically different
    """
    t_value, p_value = stats.ttest_rel(data1, data2)
    crititcal_t_value = stats.t.ppf(1 - (alpha / 2), len(data1))
    print("ttest parameters:", t_value, p_value, crititcal_t_value)

    if p_value < alpha:
        return True

    return False


def read_column_from_csv(filename, column_name):
    ret_list = []
    with open(".\\results\\" + filename) as f:
        reader = csv.reader(f)
        column_index = 0
        for i, row in enumerate(reader):
            if i == 0:
                column_index = row.index(column_name)
            else:
                ret_list.append(float(row[column_index]))
    return ret_list


file_names = []
for clustering_method in ["affinity", "c3m"]:
    for stemming_method in ["f5stemmer", "turkishstemmer", "nostemmer"]:
        for stopword_method in ["147stopwords", "nostopwords"]:
            file_names.append(clustering_method + "_" + stemming_method + "_" + stopword_method + "_results.csv")

print("File Order")
for name in file_names:
    print(name)

ttest_results = []
for i in range(0, len(file_names)):
    result_row = [None] * (i + 1)
    for j in range(i + 1, len(file_names)):
        acc_list1 = read_column_from_csv(file_names[i], "F1Score")
        acc_list2 = read_column_from_csv(file_names[j], "F1Score")
        result_row.append(statistically_different(acc_list1, acc_list2))
    ttest_results.append(result_row)

print("Pairwise Ttest for F1 scores")
print(ttest_results)

ttest_results = []
for i in range(0, len(file_names)):
    result_row = [None] * (i + 1)
    for j in range(i + 1, len(file_names)):
        acc_list1 = read_column_from_csv(file_names[i], "Runtime")
        acc_list2 = read_column_from_csv(file_names[j], "Runtime")
        result_row.append(statistically_different(acc_list1, acc_list2))
    ttest_results.append(result_row)

print("Pairwise Ttest for running time")
print(ttest_results)