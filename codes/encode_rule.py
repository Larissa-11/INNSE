import itertools
import copy
import random
import numpy as np
import json


# # 生成组合
def generate_combinations(Fragment_length):
    min_unique_elements = 3
    valid_combinations = []
    filtered_combinations = []
    seq_pool = ['A', 'T', 'C', 'G']
    combinations = list(itertools.product(seq_pool, repeat=Fragment_length))
    for combo in combinations:
        unique_elements = set(combo)
        if len(unique_elements) >= min_unique_elements:
            filtered_combinations.append(combo)
    for combo in filtered_combinations:
        if check_constraints(combo):
            valid_combinations.append(combo)
    return valid_combinations


# # 检查约束条件是否满足
def check_constraints(seq):
    # 检查 GC 含量是否为 50%
    gc_content = (seq.count('G') + seq.count('C')) / len(seq)
    if gc_content != 0.5:
        return False
    # if gc_content < 0.4 or gc_content > 0.6:
    #     return False
    #
    #     # 检查相邻元素是否满足条件
    for i in range(0, len(seq) - 1, 2):
        if (seq[i] == 'A' and seq[i + 1] == 'A') or (seq[i] == 'T' and seq[i + 1] == 'T') or (
                seq[i] == 'G' and seq[i + 1] == 'G') or (seq[i] == 'C' and seq[i + 1] == 'C'):
            return False
    return True


# # 检查汉明距离是否满足条件
def hamming_distance(seq1, seq2):
    return sum(a != b for a, b in zip(seq1, seq2))


# 检查二级结构
def filter_Secondary_sequences(sequences):
    filtered_sequences = []
    for sequence in sequences:
        complement = ""
        for base in sequence:
            if base == "A":
                complement += "T"
            elif base == "T":
                complement += "A"
            elif base == "C":
                complement += "G"
            elif base == "G":
                complement += "C"
        if complement[::-1] not in sequences:
            filtered_sequences.append(sequence)
    return filtered_sequences


# def filter_undesired_motifs(sequences):
#     filtered_sequences = []
#     for i in range(len(sequences)):
#         is_valid = True
#         sequences_str = ''.join(sequences[i])
#         if "GGC" in sequences_str or "GAATTC" in sequences_str:
#             break
#         else:
#             for j in range(len(sequences)):
#                 new_tuple = sequences[i] + sequences[j]
#                 new_str = ''.join(new_tuple)
#                 if "GGC" in new_str or "GAATTC" in new_str:
#                     is_valid = False
#                     break
#             if is_valid:
#                 filtered_sequences.append(sequences[i])
#     return filtered_sequences
def filter_undesired_motifs(sequences):
    filtered_sequences = []
    delete_index = []
    for i in range(len(sequences)):
        sequences_str = ''.join(sequences[i])
        if "GGC" in sequences_str or "GAATTC" in sequences_str:
            continue
        else:
            filtered_sequences.append(sequences[i])
    for i in range(len(filtered_sequences)):
        for j in range(len(filtered_sequences)):
            new_tuple = filtered_sequences[i] + filtered_sequences[j]
            new_str = ''.join(new_tuple)
            if "GGC" in new_str or "GAATTC" in new_str:
                delete_index.append(j)
                break
        # if is_valid:
        #     filtered_sequence.append(sequences[i])
    for i in sorted(list(set(delete_index)), reverse=True):
        del filtered_sequences[i]
    return filtered_sequences


def filter_tuples(Hamming_combinations, input_list):
    Hamming_combinations = [Hamming_combinations]
    for i in range(len(input_list)):
        is_valid = True
        for j in range(len(Hamming_combinations)):
            if hamming_distance(input_list[i], Hamming_combinations[j]) <= 3:
                is_valid = False
        if is_valid:
            Hamming_combinations.append(input_list[i])
    return Hamming_combinations


def delete_one_element(lst):
    all_combinations = []
    for i in range(len(lst)):
        new_lst = list(lst)
        del new_lst[i]
        all_combinations.append(tuple(new_lst))
    return all_combinations


def delete_check(seq1, seq2):
    combinations_seq1 = delete_one_element(seq1)
    combinations_seq2 = delete_one_element(seq2)
    for tuple1 in combinations_seq1:
        for tuple2 in combinations_seq2:
            if tuple1 == tuple2:
                return True
    return False


def find_delete_combinations(de_sequence, de_sequence_array):
    de_result = [de_sequence]
    deletecombinations = copy.copy(de_result)
    for seq in de_sequence_array:
        de_result = copy.copy(deletecombinations)
        i = 0
        for comb in de_result:
            Tf = delete_check(seq, comb)
            if Tf == False:
                i = i + 1
            if i == len(de_result):
                deletecombinations.append(seq)
    return deletecombinations


def generate_valid_combinations(Fragment_length):
    is_valid = True
    while is_valid:
        # # 生成满足条件的组合
        valid_combinations = generate_combinations(Fragment_length)
        # # 根据生成的组合，再次检查汉明距离
        Hamming_combinations = random.choice(valid_combinations)
        validHamming_combinations = filter_tuples(Hamming_combinations, valid_combinations)
        # 根据生成的组合，再次检查删除
        Delete_combinations = random.choice(validHamming_combinations)
        validDelete_combinations = find_delete_combinations(Delete_combinations, validHamming_combinations)
        filtered_Secondary_sequences = filter_Secondary_sequences(validDelete_combinations)
        filtered_undesired_motifs = filter_undesired_motifs(filtered_Secondary_sequences)
        # # 打印满足条件的组合
        if len(filtered_undesired_motifs) >= 16:
            is_valid = False
            digits = range(len(filtered_undesired_motifs[:16]))
            digits_hex = np.char.mod('%x', digits)
            result_dict = {}
            for digit, tpl in zip(digits_hex, filtered_undesired_motifs[:16]):
                result_dict[digit] = tpl
    return result_dict


Fragment_length = 6
combinations = generate_valid_combinations(Fragment_length)


def save_dict_to_file(dictionary, filepath):
    with open(filepath, 'w') as file:
        json.dump(dictionary, file)


save_path = './CodeBook/CodeBookfuben.txt'
save_dict_to_file(combinations, save_path)

print(combinations)
