"""This file holds global config details.


"""

# ******* #
# Seaborn #
# ******* #

# All configs stored in a nested dictionary 
SEABORN_CONFIG = {}

# Seaborn color/axes config
# ------------------------- #

# Predefined defaults
SEABORN_CONFIG['style'] = 'darkgrid'
SEABORN_CONFIG['context'] = {'context': 'notebook', 
                             'font_scale': 1.3, 
                             'rc': None}
SEABORN_CONFIG['pallet'] = 'deep'

# Custom seaborn style 
custom_style_dict = {
    # 'axes.axisbelow': True,
    # 'axes.edgecolor': '.8',
    # 'axes.facecolor': 'white',
    # 'axes.grid': True,
    # 'axes.labelcolor': '.15',
    # 'axes.linewidth': 1,
    # 'font.family': ['sans-serif'],
    # 'font.sans-serif': [
    #         'Arial',
    #         'Liberation Sans',
    #         'Bitstream Vera Sans',
    #         'sans-serif'
    #         ],
    # 'grid.color': '.8',
    # 'grid.linestyle': '-',
    # 'image.cmap': 'Greys',
    # 'legend.frameon': False,
    # 'legend.numpoints': 1,
    # 'legend.scatterpoints': 1,
    # 'lines.solid_capstyle': 'round',
    # 'text.color': '.15',
    # 'xtick.color': '.15',
    # 'xtick.direction': 'out',
    # 'xtick.major.size': 0,
    # 'xtick.minor.size': 0,
    # 'ytick.color': '.15',
    # 'ytick.direction': 'out',
    # 'ytick.major.size': 0,
    # 'ytick.minor.size': 0
}

SEABORN_CONFIG['custom_style'] = custom_style_dict

# Custom Matplotlib RC
custom_mpl_rc = {
    # 
}

SEABORN_CONFIG['custom_mpl_rc'] = custom_mpl_rc
