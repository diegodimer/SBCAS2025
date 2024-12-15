
import numpy as np
from functional import seq

# Code borrowed from https://github.com/aws/amazon-sagemaker-clarify/blob/53cb4172bea1efd673b6d48c3a006ce4ac1fd5a5/src/smclarify/util/__init__.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.


def pdf(xs) -> dict:
    """
    Probability distribution function.
    
    :param xs: input sequence
    :return: sequence of tuples as (value, frequency)
    """
    counts = seq(xs).map(lambda x: (x, 1)).reduce_by_key(lambda x, y: x + y)
    total = counts.map(lambda x: x[1]).sum()
    result_pdf = counts.map(lambda x: (x[0], x[1] / total)).sorted().list()
    return result_pdf


def pdfs_aligned_nonzero(*args) -> list[np.ndarray]:
    """
    Convert a list of discrete pdfs / freq counts to aligned numpy arrays of the same size for common non-zero elements.

    :return: pair of numpy arrays of the same size with the aligned pdfs.
    """
    num_pdfs = len(args)
    pdfs = []
    for x in args:
        pdfs.append(pdf(x))

    def keys(_xs):
        return seq(_xs).map(lambda x: x[0])

    # Extract union of keys
    all_keys = seq(pdfs).flat_map(keys).distinct().sorted()

    # Index all pdfs by value
    dict_pdfs = seq(pdfs).map(dict).list()

    # ADDED BY ME: if key is not in the dict_pdfs, it should be with 0 probability
    for i in all_keys:
        for j in dict_pdfs:
            if i not in j:
                j[i] = 0.0

    # result aligned lists
    aligned_lists: list[list] = [[] for x in range(num_pdfs)]

    # fill keys present in all pdfs
    # OTHER CHANGE BY ME: if key is not present, it should have 0 probability. KS results would not be precise if not
    for i, key in enumerate(all_keys):
        # for j, d in enumerate(dict_pdfs):
        #     if d.get(key, 0) == 0:
        #         break
        # else:
        # All keys exist and are != 0
        for j, d in enumerate(dict_pdfs):
            aligned_lists[j].append(d[key])
    np_arrays = seq(aligned_lists).map(np.array).list()
    return np_arrays
