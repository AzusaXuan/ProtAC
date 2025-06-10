import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
import gc
import itertools
import argparse
import utils


def get_suffixes(file_path):
    with h5py.File(file_path, 'r') as file:
        keys = file.keys()
        suffixes = set(name.split('_')[-1] for name in keys if 'seqs' in name)
    return suffixes


def read_datasets(input_dir, suffix):

    inds_name = f'inds_{suffix}'
    seqs_name = f'seqs_{suffix}'
    kws_name = f'kws_{suffix}'
    prots_name = f'prot_ids_{suffix}'
    print(suffix)

    with h5py.File(input_dir, 'r') as f:
        inds_data = f[inds_name][()].astype(np.int32)
        kws_data = list(f[kws_name])
        seqs_data = list(f[seqs_name])
        prot_ids_data = list(f[prots_name])

    return inds_data, seqs_data, kws_data, prot_ids_data


def main(num_workers, input_prefix, output_path, ranks):
    file_paths = [f"{input_prefix}_{rank}_raw.hdf5" for rank in range(ranks)]

    # Generate a list of all the file and suffix combinations
    tasks = []
    for file_path in file_paths:
        suffixes = get_suffixes(file_path)
        for suffix in suffixes:
            tasks.append((file_path, suffix))

    print("start fetching data!")
    # Create a process pool and read each group of inds, seqs, kws datasets in parallel
    with Pool(num_workers) as pool:
        results = pool.starmap(read_datasets, tasks)

    print("concating")
    # Concatenate the results of parallel reading by type
    inds_combined = np.concatenate([result[0] for result in results], axis=0)
    seqs_combined = list(itertools.chain.from_iterable(result[1] for result in results))
    kws_combined = list(itertools.chain.from_iterable(result[2] for result in results))
    prot_ids_combined = list(itertools.chain.from_iterable(result[3] for result in results))

    # Clean up memory
    del results
    gc.collect()

    print("writing")
    # Save the merged datasets to a new HDF5 file
    with h5py.File(output_path, 'a') as f:
        if 'inds' in f:
            del f['inds']
        if 'seqs' in f:
            del f['seqs']
        if 'kws' in f:
            del f['kws']
        if 'ids' in f:
            del f['ids']

        print("writing inds")
        f.create_dataset('inds', data=inds_combined)
        del inds_combined
        gc.collect()

        print("writing kws")
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        kws_dataset = f.create_dataset('kws', (len(kws_combined),), dtype=dt)
        kws_dataset[0:len(kws_combined)] = kws_combined

        print("writing seqs")
        str_dt = h5py.special_dtype(vlen=str)
        seqs_dataset = f.create_dataset('seqs', (len(seqs_combined),), dtype=str_dt)
        seqs_dataset[0:len(seqs_combined)] = seqs_combined

        print("writing ids")
        ids_dataset = f.create_dataset('prot_ids', (len(prot_ids_combined),), dtype=str_dt)
        ids_dataset[0:len(prot_ids_combined)] = prot_ids_combined


def list_datasets(hdf5_file, group_key='/', limit=1, if_print=False):
    with h5py.File(hdf5_file, 'r') as f:
        # Check if the current group/key is a dataset
        if isinstance(f[group_key], h5py.Dataset):
            print(f"Dataset: {group_key}")
            dataset = f[group_key]
            data = dataset[()]  # Read the dataset
            print(f"Data type: {dataset.dtype}")
            print(f"Data shape: {data.shape}")
            if if_print:
                print(f"First {limit} entries:")
                print(data[:limit])  # Print the first 'limit' entries
                print()  # Add an empty line for better readability
        else:
            # If the current group contains more groups or datasets, list them recursively
            for key in f[group_key].keys():
                sub_key = f"{group_key}/{key}" if group_key != '/' else f"/{key}"
                list_datasets(hdf5_file, sub_key, limit, if_print)


def convert_dataset(input_dir, output_dir, input_name, is_expand=False, is_squeeze=True):
    # Open the original HDF5 file
    with h5py.File(input_dir, 'r') as original_file:  # Use 'r' mode to open the file in read-only mode
        # Read the original dataset
        original_data = original_file[input_name][:]

        if input_name == 'inds':
            converted_data = original_data.astype(np.int32)

        if input_name == 'seqs':
            if is_expand:
                original_data = np.expand_dims(original_data, axis=1)
            if is_squeeze:
                original_data = np.squeeze(original_data, axis=1)
            converted_data = original_data.astype(np.int16)

        if input_name == 'kws':
            converted_data = original_data.astype(np.bool_)

    print("writing")
    # Create a new HDF5 file
    with h5py.File(output_dir, 'a') as new_file:  # Use 'w' mode to open the file in write mode
        # Create a new dataset in the new file and write the modified data
        new_file.create_dataset(input_name, data=converted_data)


def compress_dataset_kws(input_dir, output_dir, chunk_size=1000):
    # Open the source HDF5 file
    with h5py.File(input_dir, 'r') as input_h5:
        # Get the size of the kws dataset
        num_rows = input_h5['kws'].shape[0]

        # Open or create the target HDF5 file
        print("writing to new file")
        with h5py.File(output_dir, 'w') as output_h5:
            # Create a new dataset to store the True indices of each row, without compression
            dt = h5py.special_dtype(vlen=np.dtype('int32'))
            kws_dataset = output_h5.create_dataset('kws', (num_rows,), dtype=dt)

            # Process the kws dataset in chunks
            for i in range(0, num_rows, chunk_size):
                # Calculate the end index of the current chunk
                stop = min(i + chunk_size, num_rows)

                # Read the data of the current chunk
                bool_matrix_chunk = input_h5['kws'][i:stop]

                # Calculate the indices of True values in each row
                true_indices_per_row_chunk = [
                    np.where(row)[0].astype(np.int32) for row in bool_matrix_chunk
                ]

                # Write the True indices of each row to the HDF5 file
                kws_dataset[i:stop] = true_indices_per_row_chunk

                # Clean up memory
                del bool_matrix_chunk
                del true_indices_per_row_chunk
                gc.collect()

            print("copying other datasets")
            # Copy other datasets without compression
            for dataset_name in input_h5:
                if dataset_name != 'kws':
                    data = input_h5[dataset_name][:]
                    output_h5.create_dataset(dataset_name, data=data)


def compress_dataset_seqs(input_dir, output_dir):
    # Open the source HDF5 file
    with h5py.File(input_dir, 'r') as input_h5:
        # Read the "seqs" dataset
        seqs = input_h5['seqs'][:]
        seqs = np.squeeze(seqs, axis=1)
        converted_data = seqs.astype(np.int8)
        # Open or create the target HDF5 file
        print("writing to new file")
        with h5py.File(output_dir, 'w') as output_h5:
            output_h5.create_dataset('seqs', data=converted_data)

            # Clean up memory
            del seqs, converted_data
            gc.collect()

            print("copying to new file")
            # Copy other datasets without compression
            for dataset_name in input_h5:
                if dataset_name != 'seqs':
                    data = input_h5[dataset_name][:]
                    output_h5.create_dataset(dataset_name, data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge hdf5')
    parser.add_argument('--input_prefix', default='path/to/input_prefix', type=str,
                        help='input_prefix')
    parser.add_argument('--output_path', default='path/to/output_path', type=str,
                        help='output_path')
    parser.add_argument('--rank', default=4, type=int,
                        help='rank')

    args = parser.parse_args()
    num_processes = cpu_count()
    print("num_processes: ", num_processes)
    main(num_processes, args.input_prefix, args.output_path, args.rank)
    print("---------------output------------")
    list_datasets(args.output_path, if_print=True)