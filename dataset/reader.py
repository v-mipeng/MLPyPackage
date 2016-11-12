'''Read raw dataset from disk with pandas

We do this work based on pandas. It provides many useful and fast file parser.
The loaded dataset is converted into a RawDataset object.
'''

import pandas as pd

from pml.dataset.base import RawDataset


def reade_csv(read_from, names, sep='\t',usecols=None, **kwargs):
    '''Read raw dataset with pandas

    :param read_from: str
            File name to read data from
    :param names:list of str
            Name of each fields
    :param sep : str, default {default}
            Delimiter to use. If sep is None, will try to automatically determine
            this. Separators longer than 1 character and different from '\s+' will be
            interpreted as regular expressions, will force use of the python parsing
            engine and will ignore quotes in the data. Regex example: '\\r\\t'
    :param usecols : array-like, default None
            Return a subset of the columns. All elements in this array must either
            be positional (i.e. integer indices into the document columns) or strings
            that correspond to column names provided either by the user in `names` or
            inferred from the document header row(s). For example, a valid `usecols`
            parameter would be [0, 1, 2] or ['foo', 'bar', 'baz']. Using this parameter
            results in much faster parsing time and lower memory usage.
    :param kwargs:
            Parameter to control the reading.
            Refer pandas for more detail information
    :return: pml.dataset.base.RawDataset
            An instance of RawDataset

    --Sample--

    File: word2freq.txt
        word_one    \t      12
        word_two    \t      23
        word_three  \t      17
    dataset = pd.read_csv(filepath_or_buffer='word2freq.txt', names=['word','freq'], sep='\t', usecols=None)
            word        freq
        0   word_one    12
        1   word_two    23
        2   word_three  17
    dataset = pd.read_csv(filepath_or_buffer='word2freq.txt', names = ['word'], sep='\t', usecols=[0])
            word
        0   word_one
        1   word_two
        2   word_three
    '''
    dataset = pd.read_csv(filepath_or_buffer=read_from, names=names, sep=sep, usecols=usecols, error_bad_lines=False,
                          warn_bad_lines=True, encoding='utf-8',
                          *kwargs)
    return RawDataset(dataset.to_dict('list'))


def read_excel(*args, **kwargs):
    dataset = pd.read_excel(*args, **kwargs)
    return RawDataset(dataset.to_dict('list'))


def read_json(*args, **kwargs):
    """
    Convert a JSON string to pandas object

    Parameters
    ----------
    path_or_buf : a valid JSON string or file-like, default: None
        The string could be a URL. Valid URL schemes include http, ftp, s3, and
        file. For file URLs, a host is expected. For instance, a local file
        could be ``file://localhost/path/to/table.json``

    orient

        * `Series`

          - default is ``'index'``
          - allowed values are: ``{'split','records','index'}``
          - The Series index must be unique for orient ``'index'``.

        * `DataFrame`

          - default is ``'columns'``
          - allowed values are: {'split','records','index','columns','values'}
          - The DataFrame index must be unique for orients 'index' and
            'columns'.
          - The DataFrame columns must be unique for orients 'index',
            'columns', and 'records'.

        * The format of the JSON string

          - split : dict like
            ``{index -> [index], columns -> [columns], data -> [values]}``
          - records : list like
            ``[{column -> value}, ... , {column -> value}]``
          - index : dict like ``{index -> {column -> value}}``
          - columns : dict like ``{column -> {index -> value}}``
          - values : just the values array

    typ : type of object to recover (series or frame), default 'frame'
    dtype : boolean or dict, default True
        If True, infer dtypes, if a dict of column to dtype, then use those,
        if False, then don't infer dtypes at all, applies only to the data.
    convert_axes : boolean, default True
        Try to convert the axes to the proper dtypes.
    convert_dates : boolean, default True
        List of columns to parse for dates; If True, then try to parse
        datelike columns default is True; a column label is datelike if

        * it ends with ``'_at'``,

        * it ends with ``'_time'``,

        * it begins with ``'timestamp'``,

        * it is ``'modified'``, or

        * it is ``'date'``

    keep_default_dates : boolean, default True
        If parsing dates, then parse the default datelike columns
    numpy : boolean, default False
        Direct decoding to numpy arrays. Supports numeric data only, but
        non-numeric column and index labels are supported. Note also that the
        JSON ordering MUST be the same for each term if numpy=True.
    precise_float : boolean, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality
    date_unit : string, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.

    Returns
    -------
    result : Series or DataFrame
    """
    dataset = pd.read_json(*args, **kwargs)
    return RawDataset(dataset.to_dict('list'))
