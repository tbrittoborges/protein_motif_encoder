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
    if descriptor is None:
        descriptor = pd.Series(data.values.ravel()).unique()
        descriptor = pd.get_dummies(descriptor)
    elif hasattr(descriptor, 'get'):
        pass
    else:
        raise(ValueError("descriptor arguments should have be a dict or pandas.Series"))

    if col_names is None:
        col_names = ["{}_{}".format(col_name, letter) for letter, col_name in
                     product(descriptor.keys(), data.columns)]

    with_dummies = []
    for col in data:
        with_dummies.append(data[col].apply(descriptor.get))
    encoded = pd.concat(with_dummies, axis=1)
    encoded.columns = col_names
    return encoded


def dataframe_AAindex_encoder(data, attribute='mean', indices=None):  # aa index
    """Encode a DataFrame of Motifs to correspondent AAindex vector
    Total number of descriptors 512

    :param data:
    :param indices:
    :param attribute: panda DataFrame attribute or None
    :param motifs_dataframe: Motif DataFrame
    :return: data shape (original_n_rows, original_n_cols * n_aa_index_features)

    :Example:

    >>> import pandas as pd
    >>> df = pd.DataFrame(map(list, ['ABC', 'AAA', 'ACD']))
    >>> print(dataframe_AAindex_encoder(df).iloc[:, 0: 3])
              0         1     2
    0  4.350000  0.610000  1.18
    2  4.586667  0.713333  1.04

    .. note:: If you use AAindex please cite one of their references: http://www.genome.jp/aaindex/
    .. note:: will silently drop rows with non-standard residues symbols
    .. todo:: select a subset of aa_index
    """
    data_path = os.path.join(os.path.dirname(__file__), 'aaindex.csv')
    aa_index = pd.read_csv(data_path, index_col=(0, 1))
    aa_index = aa_index.T if not indices else aa_index.T.iloc[indices, :]

    result = []
    for i in aa_index:
        to_replace = aa_index.loc[:, i]
        replaced = data.replace(to_replace)
        # if attribute is none return full replaced df.
        result.append(getattr(replaced, attribute)(axis=1) if attribute else replaced)

    encoded = pd.concat(result, axis=1)
    if attribute is None:
        pos = range(data.shape[1]) * aa_index.shape[1]
        encoded.columns = ['{}_{}'.format(a, b) for a, b in
                           izip_longest(aa_index.columns, pos, fillvalue=aa_index.columns)]
    else:
        encoded.columns = aa_index.columns
    return encoded
