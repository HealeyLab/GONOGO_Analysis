__author__ = 'Matheus'
import matplotlib.pyplot as plt
import matplotlib as mpl
from re import search
from matplotlib.backends.backend_pdf import PdfPages
from load_data import *
import csv
from format_axes import *

from glob import glob
from re import split


def plot_daily_performance(file_names, column_to_plot, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_errors = False
    if 'plot_errors' in kwargs.keys():
        if kwargs['plot_errors']:
            plot_errors = True

    values_per_bird = list()
    for i, file_name in enumerate(file_names):
        cur_df = pd.read_excel(file_name)

        if column_to_plot not in ('hit_rate', 'rejection_rate'):
            cur_data = list(cur_df[column_to_plot])
        elif column_to_plot is 'hit_rate':
            cur_data = [hit / (hit + miss)*100 for hit, miss in zip(cur_df['hit'], cur_df['miss'])]
        elif column_to_plot is 'rejection_rate':
            cur_data = [reject / (reject + false_alarm)*100 for
                        reject, false_alarm in zip(cur_df['reject'], cur_df['false_alarm'])]
        values_per_bird.append(cur_data)
        # x_axis = np.arange(1, 15)  # plot 2 weeks
        line_style = '-o'
        cmap_multiplier = 50
        color_alpha = 0.5

        if column_to_plot is not 'trials':
            ax.axhline(50, linestyle=':', color='black')
            if column_to_plot not in ('hit_rate', 'rejection_rate'):
                ax.axhline(70, linestyle=':', color='goldenrod')

        ax.plot(np.arange(1, 1 + len(cur_data)),
                cur_data, line_style,
                color=plt.cm.jet(i * cmap_multiplier), alpha=color_alpha)

    # mean_line_color = 'red'
    # mean_line_style = '--'
    #
    # max_days = np.max([len(animal_data) for animal_data in values_per_bird])
    #
    # for animal_data in values_per_bird:
    #     while len(animal_data) < max_days:
    #         animal_data.append(np.NaN)
    #
    # means_to_plot = np.nanmean(values_per_bird, axis=0)
    # error_to_plot = list()
    # for day in range(0, len(means_to_plot)):
    #     cur_day = np.array([animal_data[day] for animal_data in values_per_bird])
    #     cur_day = cur_day[~np.isnan(cur_day)]
    #     error_to_plot.append(np.nanstd(cur_day, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(cur_day), axis=0)))
    #
    #     # cur_ci = stats.t.interval(0.95, len(cur_day) - 1, loc=np.mean(cur_day), scale=stats.sem(cur_day))
    #     # cis_to_plot.append(cur_ci[1] - cur_ci[0])
    #
    # # Plot means and CIs
    # # ax.errorbar(np.arange(1, len(means_to_plot)+1), means_to_plot, yerr=error_to_plot, color='k', zorder=100)
    # # ax.errorbar(np.arange(cur_x_start, cur_x_start + len(cur_birds_data[0])),
    # #             means_to_plot, yerr=error_to_plot, color='r', zorder=100, alpha=0.5)
    # if not plot_errors:
    #     ax.plot(np.arange(1, 1 + len(values_per_bird[0])),
    #             means_to_plot, linestyle=mean_line_style, color=mean_line_color, zorder=100, alpha=1)
    # else:
    #     ax.errorbar(np.arange(1, 1 + len(values_per_bird[0])),
    #                 means_to_plot, linestyle=mean_line_style, color=mean_line_color,
    #                 yerr=error_to_plot, zorder=100, alpha=1)

    # ax.spines['bottom'].set_visible(False)

    if 'ylim' in kwargs.keys():
        ax.set_ylim(kwargs['ylim'])
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'])
    ax.set_xlabel('Days of training')
    ax.set_xticklabels([])
    ax.set_xticks([])
    format_ax(ax)

    fig.tight_layout()

    return fig

if __name__ == "__main__":
    # label_font_size = 20
    # tick_label_size = 15
    # mpl.rcParams['figure.figsize'] = (12, 10)
    # mpl.rcParams['figure.dpi'] = 600
    # mpl.rcParams['font.size'] = label_font_size
    # mpl.rcParams['axes.labelsize'] = label_font_size
    # mpl.rcParams['axes.linewidth'] = label_font_size / 12.
    # mpl.rcParams['axes.titlesize'] = label_font_size
    # mpl.rcParams['legend.fontsize'] = label_font_size / 2.
    # mpl.rcParams['xtick.labelsize'] = tick_label_size
    # mpl.rcParams['ytick.labelsize'] = tick_label_size
    # mpl.rcParams['errorbar.capsize'] = label_font_size / 10.
    # mpl.rcParams['lines.markersize'] = label_font_size / 2.
    # mpl.rcParams['lines.linewidth'] = label_font_size / 8.

    path = ".\\Discrimination data"

    file_names = glob(path + r"\*xlsx")

    srof_file_names = [file_name for file_name in file_names if 'SROF' in file_name]
    srcf_file_names = [file_name for file_name in file_names if 'SRCF' in file_name]

    pdf_names = ['SROF_discrimination_treatment.pdf', 'SRCF_discrimination_treatment.pdf']
    plot_errors = False
    for pdf_name, files in zip(pdf_names, (srof_file_names, srcf_file_names, file_names)):
        with PdfPages(pdf_name) as pdf:
            f = plot_daily_performance(files, '%correct', ylim=[0, 100], ylabel='% Correct', plot_errors=plot_errors)
            pdf.savefig()
            plt.close(f)
            plt.clf()
            plt.cla()

            f = plot_daily_performance(files, 'hit_rate', ylim=[0, 100], ylabel='Hit rate', plot_errors=plot_errors)
            pdf.savefig()
            plt.close(f)
            plt.clf()
            plt.cla()

            f = plot_daily_performance(files, 'rejection_rate', ylim=[0, 100], ylabel='Rejection rate', plot_errors=plot_errors)
            pdf.savefig()
            plt.close(f)
            plt.clf()
            plt.cla()
            #
            # f = plot_daily_performance(files, 'd\'_loglinear_corrected', ylim=[-2.5, 2.5], ylabel='d\'', plot_errors=plot_errors)
            # pdf.savefig()
            # plt.close(f)
            # plt.clf()
            # plt.cla()
            # 
            # f = plot_daily_performance(files, 'trials', ylabel='Number of activations', plot_errors=plot_errors)
            # pdf.savefig()
            # plt.close(f)
            # plt.clf()
            # plt.cla()
