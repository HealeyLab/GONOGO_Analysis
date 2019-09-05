from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl
from re import split
from matplotlib.backends.backend_pdf import PdfPages
from load_data import *
import csv
from scipy import integrate
from format_axes import *
from matplotlib.lines import Line2D
import runpy


if __name__ == "__main__":
    label_font_size = 20
    tick_label_size = 15
    mpl.rcParams['figure.figsize'] = (12, 10)
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.size'] = label_font_size * 1.5
    mpl.rcParams['axes.labelsize'] = label_font_size * 1.5
    mpl.rcParams['axes.titlesize'] = label_font_size
    mpl.rcParams['axes.linewidth'] = label_font_size / 12.
    mpl.rcParams['legend.fontsize'] = label_font_size / 2.
    mpl.rcParams['xtick.labelsize'] = tick_label_size * 1.5
    mpl.rcParams['ytick.labelsize'] = tick_label_size * 1.5
    mpl.rcParams['errorbar.capsize'] = label_font_size
    mpl.rcParams['lines.markersize'] = label_font_size / 2.
    mpl.rcParams['lines.linewidth'] = label_font_size / 8.


    runpy.run_module(mod_name='Discrimination_logit', run_name='__main__')
    runpy.run_module(mod_name='Discrimination_daily_plots', run_name='__main__')
    runpy.run_module(mod_name='Daily_performance_treatment', run_name='__main__')
    runpy.run_module(mod_name='Logistic_regression_GO_vs_NOGO', run_name='__main__')
    runpy.run_module(mod_name='Instantaneous_performance_delta_D-1-D1', run_name='__main__')
    runpy.run_module(mod_name='ResponseBiasByTrial', run_name='__main__')
    runpy.run_module(mod_name='Trial_activation_rate', run_name='__main__')
    runpy.run_module(mod_name='Cannulation_surgery_logit', run_name='__main__')
    runpy.run_module(mod_name='Daily_performance_cannulation', run_name='__main__')
    runpy.run_module(mod_name='Trial_activation_rate', run_name='__main__')
    exit()
