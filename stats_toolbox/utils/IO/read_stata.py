""" Code to read in data from the US National Survey of Family Growth (NSFG).

This file contains modified code based on Chapter one of "Think Stats", by Allen B. 
Downey, Second Edition, available from greenteapress.com

My modifications simplify the codebase necessary to have a generalised function 
to read any Stata file and suplied Stata dictionary. These functions utilise the 
pandas methods to read from a fixed width file format.

Additional, support for calling from the command line has also been added so as 
to chain this function into more complex workflows.

All code is written and tested in Python 3, though syntax used is intened to be 
backwardly compatable with Python 2.7.

More information about the NSFG is at http://www.cdc.gov/nchs/nsfg.htm

Data can be downloaded from:
http://www.cdc.gov/nchs/data_access/ftp_dua.htm?url_redirect=ftp://ftp.cdc.gov
/pub/Health_Statistics/NCHS/Datasets/NSFG/

"""

import sys
import re
import argparse

import pandas as pd
import numpy as np


def read_stata_dct(dct_file, **options):
    """Read a Stata dictionary file.

    Details of the format can be found at:
    http://www.stata.com/manuals13/dinfilefixedformat.pdf

    Parameters
    ----------
    dct_file : string 
        Filename of dictionary file 
    options : dict  
        Any additional keyword args to pass to open()

    Returns
    -------
    stata_vars : DataFrame
        Returns table of variables with columns:
        ['start', 'end', 'type', 'name', 'fstring', 'desc']
    """
    # Define conversion between Stata and Python types
    type_map = dict(byte=int, int=int, long=int, float=float, double=float)

    var_info = []
    for line in open(dct_file, **options):
        # Match the pattern defining a column start index
        match = re.search(r'_column\(([^)]*)\)', line)
        if match:
            # Extract start index and process rest of the line
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))
            
    columns = ['start', 'type', 'name', 'fstring', 'desc']
    stata_vars = pd.DataFrame(var_info, columns=columns)

    # Fill in the end column by shifting the start column by one
    stata_vars['end'] = stata_vars.start.shift(-1)
    stata_vars.loc[len(stata_vars)-1, 'end'] = 0
    stata_vars = stata_vars[['start', 'end', 'type', 'name', 'fstring', 'desc']]

    # Adjust index and 
    return stata_vars


def read_stata_fwf(fwf_filename, dct_filename, index_base=0, file_options=None, 
                   pandas_options=None):
    """Read in a Stata fixed width file to a pandas DataFrame.

    Details of the dct file format can be found at:
    http://www.stata.com/manuals13/dinfilefixedformat.pdf

    Parameters
    ----------
    fwf_filename : str
        Filename of fixed width file to be read in.
    dct_filename : str
        Filename of dictionary file specifying column positions 
    index_base : int
        Specifies whether to use 0 or 1 based indexing
    file_options : dict 
        Any additional keyword args to pass to `open()`
    pandas_options : dict 
        Any additional keyword args to pass to `pandas.read_fwf()`

    Returns
    -------
    Pandas DataFrame
    """
    if not file_options:
        file_options = {}
    if not pandas_options:
        pandas_options = {}

    stata_vars = read_stata_dct(dct_filename, **file_options)

    # Adjust for base index and convert to pairs of ints
    colspecs = stata_vars[['start', 'end']] - index_base
    colspecs = colspecs.astype(np.int).values.tolist()
    names = stata_vars['name']

    # Parse file using pandas, handeling compressed files
    if fwf_filename.endswith('gz'):
        compression = 'gzip'
    elif fwf_filename.endswith('bz2'):
        compression = 'bz2'
    else:
        compression = None

    df = pd.read_fwf(fwf_filename, colspecs=colspecs, 
                     names=names, compression=compression, **pandas_options)
    return df


def main(args):
    """ Write pandas DataFrame to stdout."""
    df = read_stata_fwf(fwf_filename=args.fwf_filename, dct_filename=args.dct_filename,
                        index_base=1)
    df.to_csv(sys.stdout)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "fwf_filename",
        help="Name of fixed width file to read from.")
    parser.add_argument(
        "dct_filename",
        help="Name of stata dictionary file defining column positions.")

    args = parser.parse_args()

    main(args)
