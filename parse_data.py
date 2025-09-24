
import os
import numpy as np
import pickle

def parse_gt_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            data.append([float(x) for x in line])
    return data

def main():
    data_dir = 'data/MOT16/train'
    seqs = [s for s in os.listdir(data_dir) if not s.startswith('.')]
    
    all_data = {}
    for seq in seqs:
        seq_dir = os.path.join(data_dir, seq)
        gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
        
        if os.path.exists(gt_file):
            all_data[seq] = parse_gt_file(gt_file)

    with open('data/mot16_annotations.pkl', 'wb') as f:
        pickle.dump(all_data, f)

if __name__ == '__main__':
    main()
