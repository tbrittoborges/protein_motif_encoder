# -*- coding: utf-8 -*-
import os
from itertools import product, izip_longest

import pandas as pd


def dataframe_sparse_encoder(data, descriptor=None, col_names=None):
    """Encode a pandas DataFrame with a descriptor. Similar to one hot encoding,
    however it preserves categories across columns.

    :param dict descriptor: dict-like descriptor to be applied to the columns
    :param pd.DataFrame data: DataFrame with categorical data
    :param col_names: names for the new columns
    :type col_names: list of [str,]
    :return: table with encoded data
    :rtype : pandas.DataFrame

    :Example:

        >>> import pandas as pd
        >>> df = pd.DataFrame(map(list, ['ABC', 'AAA', 'ACD']))
        >>> print(dataframe_sparse_encoder(df))
           0_A  1_A  2_A  0_B  1_B  2_B  0_C  1_C  2_C  0_D  1_D  2_D
        0    1    0    0    0    0    1    0    0    0    0    1    0
        1    1    0    0    0    1    0    0    0    1    0    0    0
        2    1    0    0    0    0    0    1    0    0    0    0    1

     .. note:: descriptor parameters can be any Python object with .get method, as pd.Series.
    """
    data = data.copy()
    if descriptor is None:
        descriptor = pd.Series(data.values.ravel()).unique()
        descriptor = pd.get_dummies(descriptor)
    elif hasattr(descriptor, 'get'):
        pass
    else:
        raise (ValueError("descriptor arguments should have be a dict or pandas.Series"))

    if col_names is None:
        col_names = ["{}_{}".format(col_name, letter) for letter, col_name in
                     product(descriptor.keys(), data.columns)]

    with_dummies = []
    for col in data:
        with_dummies.append(data[col].apply(descriptor.get))
    encoded = pd.concat(with_dummies, axis=1)
    encoded.columns = col_names
    return encoded


def dataframe_aaindex_encoder(data, indices=None, contains=None):
    """Encode a DataFrame of Motifs to correspondent AAindex vector
    Total number of descriptors 512

    :param indices:
    :param contains:
    :param data:
    :return:

    :Example:

    >>> import pandas as pd
    >>> df = pd.DataFrame(map(list, ['ABC', 'AAA', 'ACD']))
    >>> print(dataframe_aaindex_encoder(df).iloc[:, 0: 3])
              0         1     2
    0  4.350000  0.610000  1.18
    2  4.586667  0.713333  1.04

    .. note:: If you use AAindex please cite one of their references: http://www.genome.jp/aaindex/
    .. warning:: Function will silently drop rows with non-standard residues symbols
    .. todo:: select a subset of aa_index
    """
    data = data.copy()
    aa_index_path = os.path.join(os.path.dirname(__file__), 'aaindex.csv')
    aa_index = pd.read_csv(aa_index_path, index_col=(0, 1))
    # todo: limit descriptors with regex

    if indices:
        aa_index = aa_index.T.iloc[indices, :]
    elif contains:
        index = aa_index.index.get_level_values('name')
        mask = index.str.contains(contains, case=False)
        aa_index = aa_index.loc[mask].T
    else:
        aa_index = aa_index.T

    result = []
    for i in aa_index:
        to_replace = aa_index.loc[:, i]
        replaced = data.replace(to_replace)
        # if attribute is none return full replaced df.
        result.append(replaced)

    encoded = pd.concat(result, axis=1)
    pos = range(data.shape[1]) * aa_index.shape[1]
    encoded.columns = ['{}_{}'.format(a, b) for a, b in
                       izip_longest(aa_index.columns, pos, fillvalue=aa_index.columns)]

    return encoded
