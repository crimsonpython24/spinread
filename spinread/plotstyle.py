import matplotlib

# default color cycle and default greys, named once so every figure uses the same ones
BLUE, ORANGE, GRAY, DARK = "C0", "C1", "0.6", "black"
FONT = 7.0  # axis labels, tick labels, legends, text inside panels
TITLE = 8.0  # panel titles
DPI = 250
COLUMN_WIDTH = 3.25  # inches, one ICML column

RC = {
    "font.size": FONT,
    "axes.titlesize": TITLE,
    "axes.labelsize": FONT,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
    "legend.fontsize": FONT,
    "legend.frameon": False,
    "savefig.dpi": DPI,
}


def apply():
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(RC)


def finish(figure, path, **adjust):
    figure.subplots_adjust(**adjust) if adjust else figure.tight_layout(pad=0.3)
    figure.savefig(path, dpi=DPI)
