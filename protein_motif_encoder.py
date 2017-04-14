# -*- coding: utf-8 -*-
"""
Set of encoding strategies for protein motifs sequences training with machine learning.
"""
import os
from itertools import product, izip_longest

import pandas as pd
''
__all__ = ['dataframe_sparse_encoder', 'dataframe_aaindex_encoder ']


def dataframe_sparse_encoder(data, descriptor=None, col_names=None):
    """Sparsely encode the motifs with abstract descriptor. Similar to 'one-hot' encoding that 
    is consistent across the motif columns.

    :param descriptor: dict-like descriptor to be applied to the columns
    :type descriptor: dict or None
    :param pd.DataFrame data: DataFrame with the data
    :param col_names: names for the new columns
    :type col_names: list[str] or None
    :return: table with encoded data
    :rtype : pd.DataFrame
    :raise ValueError: if the descriptor argument lack `get` method


    :Example:

        >>> import pandas as pd
        >>> df = pd.DataFrame(map(list, ['ABC', 'AAA', 'ACD']))
        >>> print(dataframe_sparse_encoder(df))  # doctest: +NORMALIZE_WHITESPACE
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
        raise (ValueError("Descriptor should have be a dict or pandas.Series"))

    if col_names is None:
        col_names = ["{}_{}".format(col_name, letter) for letter, col_name in
                     product(descriptor.keys(), data.columns)]

    with_dummies = []
    for col in data:
        with_dummies.append(data[col].apply(descriptor.get))
    encoded = pd.concat(with_dummies, axis=1)
    encoded.columns = col_names
    return encoded


def dataframe_aaindex_encoder(data, indices=None, contains=None, named_columns=False):
    """Encode the motifs with an a DataFrame of Motifs to correspondent AAindex vector
    Total number of descriptors 512

    :param pd.DataFrame data: DataFrame with categorical data
    :param bool named_columns: 
    :param indices:
    :param contains:
    :return pd.DataFrame:

    :Example:

    >>> import pandas as pd; df = pd.DataFrame(map(list, ['ABC', 'AAA', 'ACD'])) 
    >>> print(dataframe_aaindex_encoder(df).iloc[:, 0: 3])  # doctest: +NORMALIZE_WHITESPACE
        0     1     2
    0  4.35     B  4.65
    1  4.35  4.35  4.35
    2  4.35  4.65  4.76


    .. note:: If you use AAindex please cite one of their references: http://www.genome.jp/aaindex/
    .. warning:: It does not replace non-standard residues symbols 
    .. todo:: subset aa_index (with regex)
    """
    data = data.copy()
    aa_index_path = os.path.join(os.path.dirname(__file__), 'aaindex.csv')
    aa_index = pd.read_csv(aa_index_path, index_col=(0, 1))
    aa_index = aa_index.T

    result = []
    for i in aa_index:
        to_replace = aa_index.loc[:, i]
        replaced = data.replace(to_replace)
        result.append(replaced)

    encoded = pd.concat(result, axis=1)
    pos = range(data.shape[1]) * aa_index.shape[1]
    # encoded.columns = ['{}_{}'.format(a, b) for a, b in
    #                    izip_longest(aa_index.columns, pos, fillvalue=aa_index.columns)]

    return encoded

