from data import data_handle
import random

input_path = "./results/comic/comic.dna"
output_path = "results/error/errorcomicindel/1.0/" + "comic"
nucleotide_insertion = 0.005
nucleotide_mutation = 0
nucleotide_deletion = 0.005
iterations = 10
dna_sequences = data_handle.read_dna_file(input_path)
for iteration in range(iterations):
    total_indices = [sequence_index for sequence_index in range(len(dna_sequences))]
    # insertion errors
    for insertion_iteration in range(int(len(dna_sequences) * nucleotide_insertion)):
        chosen_index = random.choice(total_indices)
        dna_sequences[chosen_index].insert(random.randint(0, len(dna_sequences[chosen_index]) - 1),
                                           random.choice(['A', 'C', 'G', 'T']))

    # mutation errors
    for mutation_iteration in range(int(len(dna_sequences) * nucleotide_mutation)):
        chosen_index = random.choice(total_indices)
        chosen_index_in_sequence = random.randint(0, len(dna_sequences[chosen_index]) - 1)
        chosen_nucleotide = dna_sequences[chosen_index][chosen_index_in_sequence]
        dna_sequences[chosen_index][chosen_index_in_sequence] = \
            random.choice(list(filter(lambda nucleotide: nucleotide != chosen_nucleotide,
                                      ['A', 'C', 'G', 'T'])))

    # deletion errors
    for deletion_iteration in range(int(len(dna_sequences) * nucleotide_deletion)):
        chosen_index = random.choice(total_indices)
        del dna_sequences[chosen_index][random.randint(0, len(dna_sequences[chosen_index]) - 1)]
    str_i = str(iteration)
    output = output_path + str_i + '.dna'
    data_handle.write_dna_file(output, dna_sequences)