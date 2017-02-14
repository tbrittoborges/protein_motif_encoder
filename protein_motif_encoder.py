# -*- coding: utf-8 -*-

from itertools import product

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
    encoded_data = pd.concat(with_dummies, axis=1)
    encoded_data.columns = col_names
    return encoded_data
