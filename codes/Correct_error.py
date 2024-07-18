from data import data_handle
import itertools
import math
from data.data_handle import load_dict_from_file

input_path = "./results/error/errorcomic/2.0/comic0.dna"
output_path = "./results/Error_correction/comic/2.0/2.0"

save_path = './CodeBook/CodeBook.txt'

dna_set = load_dict_from_file(save_path)
dna_set = [tuple(value) for value in dna_set.values()]


def combine_tuples(dna_set, num_combinations):
    combination_set = set()

    for tup in itertools.product(dna_set, repeat=num_combinations):
        combined = sum(tup, ())
        combination_set.add(combined)

    return combination_set


def levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def hamming_distance(seq1, seq2):
    return sum(a != b for a, b in zip(seq1, seq2))


dna_sequences = data_handle.read_dna_file(input_path)
m = len(dna_sequences)
r = []
for i in range(0, m, 1):
    r.append(len(dna_sequences[i]))
row = max(r, key=r.count)
min_element_record = {}
for i in range(len(dna_sequences)):
    for j in range(row // 6):
        start = j * 6
        end = (j + 1) * 6
        if start > len(dna_sequences[i]):
            break
        if end > len(dna_sequences[i]):
            end = len(dna_sequences[i])

        sub_sequence = tuple(dna_sequences[i][start:end])
        if sub_sequence not in dna_set:
            if start < row - 12:
                next_sub_sequence = tuple(dna_sequences[i][(j + 1) * 6: (j + 2) * 6])
                # if next_sub_sequence in dna_set and tuple(dna_sequences[i][(j + 1) * 6 : (j + 2) * 6 ]) in dna_set and tuple(
                #         dna_sequences[i][(j + 2) * 6: (j + 3) * 6]) in dna_set:
                if next_sub_sequence in dna_set and tuple(dna_sequences[i][(j + 1) * 6: (j + 2) * 6]) in dna_set:
                    min_distance = float('inf')
                    min_element = None
                    for element in dna_set:
                        distance = levenshtein_distance(sub_sequence, element)
                        if distance < min_distance:
                            min_distance = distance
                            min_element = element
                    # for element in dna_set:
                    #     distance = levenshtein_distance(sub_sequence, element)
                    #     if distance == min_distance:
                    #         min_distance = distance
                    #         l = (element)
                    #         if element == min_element:
                    #             min_element_record = {}
                    #         else:
                    #             min_element_record = {(i, start, l): element}
                    min_element_list = list(min_element)
                    dna_sequences[i][j * 6: (j + 1) * 6] = min_element_list

                elif tuple(dna_sequences[i][(j + 1) * 6 - 1: (j + 2) * 6 - 1]) in dna_set and tuple(
                        dna_sequences[i][(j + 2) * 6 - 1: (j + 3) * 6 - 1]) in dna_set and tuple(
                    dna_sequences[i][(j + 3) * 6 - 1: (j + 4) * 6 - 1]) in dna_set:
                    forward_sub_sequence = tuple(dna_sequences[i][start:end - 1])
                    min_distance = float('inf')
                    min_element = None
                    for element in dna_set:
                        distance = levenshtein_distance(forward_sub_sequence, element)
                        if distance < min_distance:
                            min_distance = distance
                            min_element = element
                    for element in dna_set:
                        distance = levenshtein_distance(forward_sub_sequence, element)
                        if distance == min_distance:
                            min_distance = distance
                            l = len(element)
                            if element == min_element:
                                min_element_record = {}
                            else:
                                min_element_record = {(i, start, l): element}

                    min_element_list = list(min_element)
                    dna_sequences[i] = dna_sequences[i][0:start] + min_element_list + dna_sequences[i][
                                                                                      end - 1:len(dna_sequences[i])]
                elif tuple(dna_sequences[i][(j + 1) * 6 + 1: (j + 2) * 6 + 1]) in dna_set and tuple(
                        dna_sequences[i][(j + 2) * 6 + 1: (j + 3) * 6 + 1]) in dna_set and tuple(
                    dna_sequences[i][(j + 3) * 6 + 1: (j + 4) * 6 + 1]) in dna_set:
                    forward_sub_sequence = tuple(dna_sequences[i][start:end + 1])
                    min_distance = float('inf')
                    min_element = None
                    for element in dna_set:
                        distance = levenshtein_distance(forward_sub_sequence, element)
                        if distance < min_distance:
                            min_distance = distance
                            min_element = element
                    for element in dna_set:
                        distance = levenshtein_distance(forward_sub_sequence, element)
                        if distance == min_distance:
                            min_distance = distance
                            l = len(element)
                            if element == min_element:
                                min_element_record = {}
                            else:
                                min_element_record = {(i, start, l): element}
                    min_element_list = list(min_element)
                    # print(dna_sequences[i][end + 1:len(dna_sequences[i])])
                    dna_sequences[i] = dna_sequences[i][0:start] + min_element_list + dna_sequences[i][
                                                                                      end + 1:len(dna_sequences[i])]
                else:
                    Remaining_sequence = tuple(dna_sequences[i][start:len(dna_sequences[i])])
                    min_distance = float('inf')
                    nearest_tuple = None
                    T = False
                    for m in range(len(Remaining_sequence) - 6 + 1):
                        substring = Remaining_sequence[m:m + 6]
                        for dna_tuple in dna_set:
                            distance = hamming_distance(list(substring), list(dna_tuple))
                            if distance < min_distance:
                                min_distance = distance
                                nearest_tuple = dna_tuple
                                # if min_distance == 0 and tuple(Remaining_sequence[m+6:m + 12]) in dna_set:
                                if min_distance == 0:
                                    T = True
                                    break
                        if T:
                            break
                    error_sequences = Remaining_sequence[0:m]
                    if len(error_sequences) == 1:
                        dna_sequences[i] = dna_sequences[i][0:start] + dna_sequences[i][start + 1:]
                    else:
                        # num_combinations = math.floor(m / 6)
                        if m <= 8:
                            num_combinations = 1
                        else:
                            num_combinations = math.ceil(m / 6)
                        result = combine_tuples(dna_set, num_combinations)
                        min_distance = float('inf')
                        min_element = None
                        for element in result:
                            distance = levenshtein_distance(error_sequences, element)
                            if distance < min_distance:
                                min_distance = distance
                                min_element = element
                        for element in result:
                            distance = levenshtein_distance(error_sequences, element)
                            if distance == min_distance:
                                min_distance = distance
                                l = len(element)
                                if element == min_element:
                                    min_element_record = {}
                                else:
                                    min_element_record = {(i, start, l): element}
                        min_element_list = list(min_element)
                        Remaining_sequence = list(Remaining_sequence[m:len(Remaining_sequence)])
                        dna_sequences[i] = dna_sequences[i][0:start] + min_element_list + Remaining_sequence
                        # break
            else:
                inter_sub_sequence = tuple(dna_sequences[i][start:len(dna_sequences[i])])
                if len(dna_sequences[i]) - start <= 9:
                    num_combinations = math.ceil(1)
                else:
                    num_combinations = math.ceil(2)
                result = combine_tuples(dna_set, num_combinations)
                min_distance = float('inf')
                min_element = None
                for element in result:
                    distance = levenshtein_distance(inter_sub_sequence, element)
                    if distance < min_distance:
                        min_distance = distance
                        min_element = element
                min_element_list = list(min_element)
                dna_sequences[i] = dna_sequences[i][0:start] + min_element_list
                # break
    if len(dna_sequences[i]) != row:
        dna_sequences[i] = dna_sequences[i][:row]
path = output_path + ".dna"
data_handle.write_dna_file(path, dna_sequences)
for r in range(len(min_element_record)):
    copied_list = list(dna_sequences)
    keys = list(min_element_record.keys())
    key = keys[r]
    min_list = list(min_element_record.values())
    min_list = [item for tpl in min_list for item in tpl]
    copied_list[key[0]] = copied_list[key[0]][0:key[1]] + min_list + copied_list[key[0]][key[1] + key[2]:]
    i_str = str(r + 1)
    path = output_path + i_str + ".dna"
    data_handle.write_dna_file(path, copied_list)
# data_handle.write_dna_file(output_path, dna_sequences)
