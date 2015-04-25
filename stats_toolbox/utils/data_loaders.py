"""Shorcut functions to load in and clean specific datasets."""

import os
from .IO.read_stata import read_stata_fwf
from .cleaners import clean_fem_preg


def load_fem_preg_2002(data_dir=None):
    """Load and preprocess pregnant female data from 2002."""
    fwf_fname = '2002FemPreg.dat.gz'
    dct_fname = '2002FemPreg.dct'
    
    if data_dir is None:
        data_dir = '../data'
    
    df = read_stata_fwf(fwf_filename=os.path.join(data_dir, fwf_fname),
                        dct_filename=os.path.join(data_dir, dct_fname),
                        index_base=1, 
                        file_options={'encoding': 'latin1'}) 
    
    df = clean_fem_preg(df)
    df = df[['caseid', 'prglngth', 'outcome', 'pregordr', 
             'birthord', 'birthwgt_lb', 'birthwgt_oz', 'agepreg', 'finalwgt']]
    
    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0
    df['birthwgt_kg'] = df['totalwgt_lb'] * 0.453592

    return df
