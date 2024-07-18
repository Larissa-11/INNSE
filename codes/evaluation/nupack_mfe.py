from nupack import *
import os


DFT_CONCERNTRATION = 1e-1
MODEL = Model(material='dna', celsius=37, sodium=1.0, magnesium=0.0, ensemble='stacking')
file = '../coded_image/Mona Lisa.dna'
score_all = []
dna_sequences = []
with open(os.path.join(file), "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()  # 去除行首尾的空白符
        if line:  # 检查是否为空白行
            dna_sequences.append(list(line))
    for i in dna_sequences:
        seq = i
        seq = "".join(seq)
        seq1 = Strand(seq, name="a")
        my_complex = Complex([seq1], name="b")
        tube1 = Tube({seq1: DFT_CONCERNTRATION}, complexes=SetSpec(max_size=1, include=[my_complex]), name="tube1")
        single_results = tube_analysis([tube1], model=MODEL, compute=['pfunc', 'pairs', 'mfe', 'sample', 'subopt'],
                                       options={'num_sample': 2, 'energy_gap': 0.5})
        score = single_results[my_complex].mfe[0].energy
        score_all.append(score)
save_file = './MFE/Mona Lisa/Mona Lisa.mfe'
with open(os.path.join(save_file), "w") as file:
    for row in range(len(score_all)):
        file.write(str(score_all[row]) + "\n")

