from numpy import fromfile, uint8
from data import data_handle


def read_bits_from_file(path, need_logs=True):
    if need_logs:
        print("Read binary matrix from file: " + path)
    values = fromfile(file=path, dtype=uint8)
    if need_logs:
        print("There are " + str(len(values) * 8) + " bits in the inputted file. ")
    return len(values) * 8


# def read_file(path):
#     data = []
#     with open(path, "r") as file:
#         # Read current file by line
#         lines = file.readlines()
#         for index, line in enumerate(lines):
#             data.append(list(line.replace("\n", "")))
#     return data


def GC_content(dna_sequences, a):
    gc_all = []
    for i in range(a):
        g = dna_sequences[i].count('G')
        c = dna_sequences[i].count('C')
        seq_len = len(dna_sequences[i])
        gc_content = (g + c) / seq_len
        gc_all.append(gc_content)
    max_gc = max(gc_all)
    min_gc = min(gc_all)
    return max_gc, min_gc


#
#
def motifs(sequence, motifs_count):
    motifs = ["GGC", "GAATTC"]
    for missing_segment in motifs:
        if missing_segment in "".join(sequence):
            motifs_count = "".join(sequence).count(missing_segment) + motifs_count
    return motifs_count


def find_longest_homopolymer(sequences):
    max_length = 1
    longest_homopolymer = ''

    for sequence in sequences:
        current_length = 1
        # current_homopolymer = sequence[0]

        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                current_length += 1
                if current_length > max_length:
                    max_length = current_length
                    longest_homopolymer = sequence[i - max_length + 1:i + 1]
            else:
                current_length = 1

    return max_length, longest_homopolymer


#
#
def calculate_percentage(data):
    count = 0
    for num in data:
        num = "".join(num)
        if float(num) > -30.00:
            count += 1
    percentage = (count / len(data)) * 100
    return percentage


if __name__ == "__main__":
    img_path = '../datasets/0801/0801.png'
    DNA_path = './code/DNA-QLC.dna'
    MFE_path = './MFE/DNA-QLC.mfe'
    bit_size = read_bits_from_file(img_path)
    dna_sequences = data_handle.read_dna_file(DNA_path)
    a = len(dna_sequences)
    b = len(dna_sequences[0])
    nt_size = a * b
    print("Number of oligos:", a)
    print("Net information density:%.2f" % (bit_size / nt_size))

    max_gc, min_gc = GC_content(dna_sequences, a)
    print('The minimum GC content is :%.0f%%' % (min_gc * 100))
    print('The maximum GC content is :%.0f%%' % (max_gc * 100))

    motifs_count = 0
    for i in range(a):
        motifs_count = motifs(dna_sequences[i], motifs_count)

    print('The motifs_count is :%d' % (motifs_count))

    max_length, max_polymer = find_longest_homopolymer(dna_sequences)
    print("The homopolymer is：", max_polymer)
    print("The length of the homopolymer：", max_length)

    dna_MFE = data_handle.read_dna_file(MFE_path)
    result = calculate_percentage(dna_MFE)
    print(f"The percentage of numbers greater than -30 is: {result:.2f}%")
