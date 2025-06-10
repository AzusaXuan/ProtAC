import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
import gc
import itertools
import argparse
import utils
import os

def main(input_prefix, output_path, ranks):
    file_paths = [f"{input_prefix}_{rank}.hdf5" for rank in range(ranks)]

    # Generate a list of all the file and suffix combinations
    total_caption_inds = []
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as caption_list:
            total_caption_inds.extend(caption_list["caption_inds"])
    print("total caption inds: ", len(total_caption_inds))
    with h5py.File(output_path, 'w') as output_h5:
        output_h5.create_dataset('caption_inds', data=total_caption_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge hdf5')
    parser.add_argument('--input_prefix', default='', type=str,
                        help='input_prefix')
    parser.add_argument('--output_path', default='', type=str,
                        help='output_path')
    parser.add_argument('--rank', default=8, type=int,
                        help='rank')

    args = parser.parse_args()
    if os.path.isfile(args.output_path):
        os.remove(args.output_path)
    main(args.input_prefix, args.output_path, args.rank)