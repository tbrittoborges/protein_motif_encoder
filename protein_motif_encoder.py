# -*- coding: utf-8 -*-
"""
Set of encoding strategies for protein motifs sequences training with machine learning.
"""
import os
from itertools import product, izip_longest, islice, repeat

import pandas as pd
from Bio import SeqIO

__all__ = ['read_fasta', 'dataframe_sparse_encoder', 'dataframe_aaindex_encoder', 'window',
           'dataframe_k_mer_composition']


def read_fasta(fasta_file):
    """
    Parses a fasta file with BioPython to a dictionary 
    
    :param fasta_file: input file in the fasta format 
    :return dict: dictionary with [identifier, sequence] from the input file 
    """
    with open(fasta_file) as open_input:
        seqs = SeqIO.to_dict(SeqIO.parse(open_input, 'fasta'),
                             key_function=lambda x: x.id.split()[0])
    return seqs


def read_positions(positions_file, columns=None):
    """
    Position for motifs within the protein sequences.

    :param positions_file: csv file with the positions
    :param array-like columns: columns with identifier and position, respectively 
    :return dict : 
    """
    if columns is None:
        columns = [0, 1]
    data = pd.read_csv(positions_file)
    try:
        data = data.loc[:, columns]
    except KeyError:  # named columns
        data = data.iloc[:, columns]

    return data.to_records()


def create_motifs_from_residues(sequence, motif_size=5, residues=['S', 'T'], extremity='X'):
    """
    Fragments the proteins in modifiable peptides.
    :param str sequence: protein sequence
    :param int motif_size: number of residues per motif
    :param srt extremity: character symbol for termination character
    :param array_like residues: residues' position within the protein sequences
    :rtype : Iterator of position collection.Iterable[tuple]
    
    .. example:
    >>> print(list(create_motifs_from_residues('VTIQHPWFKRTLGP'))) # doctest: +NORMALIZE_WHITESPACE
    [(2, 'XVTIQ'), (11, 'KRTLG')]    
    """
    if len(extremity) > 1:
        extremity = extremity[0]
    if isinstance(residues, str):
        residues = [residues,]
    elif not hasattr(residues, "__contains__"):
        raise TypeError("residue paramenter should be a list-like object")

    if motif_size & 1:  # is odd
        start = motif_size // 2
        stop = (motif_size // 2) + 1
    else:
        start = motif_size // 2
        stop = start

    protein_len = len(sequence)
    for i, aa in enumerate(sequence):
        if aa in residues:
            if i - start < 0:
                motif = sequence[0: i].rjust(start, extremity) + sequence[i: i + stop]
            elif i + stop > protein_len:
                motif = sequence[i - start: i] + sequence[i: i + stop].ljust(stop, extremity)
            else:
                motif = sequence[i - start: i + stop]

            yield i + 1, motif




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


    .. example:

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


def dataframe_aaindex_encoder(data, contains=None, named_columns=False):
    """Encode the motifs with an a DataFrame of Motifs to correspondent AAindex vector
    Total number of descriptors 512.

    :param pd.DataFrame data: DataFrame with categorical data
    :param bool named_columns: Weather output dataframe has column_names 
    :param str contains: Terms for reducing aaindex
    :return: table with encoded data
    :rtype: pd.DataFrame

    :Example:

    >>> import pandas as pd; df = pd.DataFrame(map(list, ['ABC', 'AAA', 'ACD'])) 
    >>> print(dataframe_aaindex_encoder(df).iloc[:, 0: 3])  # doctest: +NORMALIZE_WHITESPACE
        0     1     2
    0  4.35     B  4.65
    1  4.35  4.35  4.35
    2  4.35  4.65  4.76
    >>> print(dataframe_aaindex_encoder(df, contains='Hydrophobicity_index', \
    named_columns=True).iloc[:, 0:3])  # doctest: +NORMALIZE_WHITESPACE    
        0_Hydrophobicity_index_(Argos_et_al.,_1982) \  
    0                                         0.61     
    1                                         0.61     
    2                                         0.61     
    <BLANKLINE>
        1_Hydrophobicity_index_(Argos_et_al.,_1982) \  
    0                                           B   
    1                                        0.61   
    2                                        1.07   
    <BLANKLINE>
        2_Hydrophobicity_index_(Argos_et_al.,_1982) 
    0                                         1.07  
    1                                         0.61  
    2                                         0.46  
    
    .. note:: If you use AAindex please cite one of their references: http://www.genome.jp/aaindex/
    .. warning:: It does not replace non-standard residues symbols 
    .. todo:: other strategies to subset aa_index 
    """
    data = data.copy()
    aa_index_path = os.path.join(os.path.dirname(__file__), 'aaindex.csv')
    aa_index = pd.read_csv(aa_index_path, index_col=(0, 1))
    aa_index = aa_index.T

    if contains:  # subset the aaindex based on the term
        names = aa_index.columns.get_level_values('name')
        mask = names.str.contains(contains, case=False)
        aa_index = aa_index.loc[:, mask]

    result = []
    for i in aa_index:
        to_replace = aa_index.loc[:, i]
        replaced = data.replace(to_replace)
        result.append(replaced)
    encoded = pd.concat(result, axis=1)

    if named_columns:
        pos = range(data.shape[1]) * aa_index.shape[1]
        names = [x for x in aa_index.columns.get_level_values('name')
                 for _ in range(data.shape[1])]
        encoded.columns = ['{}_{}'.format(a, b) for a, b in zip(pos, names)]

    return encoded


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    :param seq: sequence to by cutted 
    :type seq:  array-like or str
    :param n: motif size 
    :return iterator: 

    ..note:
    modified from https://docs.python.org/release/2.3.5/lib/itertools-example.html
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield "".join(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield "".join(result)


def dataframe_k_mer_composition(data, k=3, alphabet=None):
    """
    Return the k-mer composition for each sample (row) in the data.

    :param motif_df: motif in the format one residue per column
    :type motif_df: pandas.DataFrame
    :param k: length of the kmer
    :type k: int
    :param alphabet: all amino acids
    :type alphabet: str
    :return: matrix with counts of k-mer
    
    :Example:
    >>> import pandas as pd; df = pd.DataFrame(map(list, ['ABC', 'AAA', 'ACD'])) 
    >>> print(dataframe_k_mer_composition(df, k=1))  # doctest: +NORMALIZE_WHITESPACE
        A_comp    B_comp    C_comp    D_comp
    0  0.333333  0.333333  0.333333  0.000000
    1  1.000000  0.000000  0.000000  0.000000
    2  0.333333  0.000000  0.333333  0.333333
    >>> print(dataframe_k_mer_composition(df, k=2).iloc[:, :3])  # doctest: +NORMALIZE_WHITESPACE
        AA_comp   AB_comp   AC_comp
    0  0.000000  0.333333  0.000000
    1  0.666667  0.000000  0.000000
    2  0.000000  0.000000  0.333333
"""
    if k == 1:  # amino acid composition
        aa_comp = data.apply(pd.value_counts, axis=1).fillna(0)
        aa_comp /= data.shape[1]
        aa_comp.columns = aa_comp.columns + "_comp"

        return aa_comp

    if alphabet is None:
        alphabet = "".join(pd.unique(data.unstack().values))

    k_mers = data.apply(lambda x: list(window(x, n=k)), axis=1)
    k_mers = k_mers.apply(pd.value_counts)

    all_kmers = ["".join(aa) for aa in product(*repeat(list(alphabet), k))]
    k_mers = k_mers.loc[:, all_kmers].fillna(0)
    k_mers /= data.shape[1]
    k_mers.columns = k_mers.columns + "_comp"

    return k_mers


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
