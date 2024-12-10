import matplotlib as mpl
from matplotlib import cycler

colors = cycler( color=["black", "003DFD", "b512b8", "11a9ba", "0d780f", "f77f07", "ba0f0f"] )
mplstyle = { 
            "font.family": "sans serif", 
            "axes.edgecolor": "black",
            "axes.grid": True, "axes.labelcolor": "#333333",
            "axes.labelweight": 600, "axes.linewidth": 1,
            "axes.prop_cycle": colors, "axes.spines.top": False,
            "axes.spines.right": False, "axes.spines.bottom": False,
            "axes.spines.left": False,
            "grid.color": "#dedede", "legend.frameon": False,
            "lines.linewidth": 1.3,
            "xtick.color": "#333333",
            "xtick.labelsize": "small",
            "ytick.color": "#333333",
            "ytick.labelsize": "small",
            "xtick.bottom": False,
            "figure.figsize": (24, 6),
            "figure.titlesize": 18,
        } 
mpl.rcParams.update(mplstyle)