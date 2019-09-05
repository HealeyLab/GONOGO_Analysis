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


def plot_daily_treatment(file_names, column_to_plot, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_flag = False
    if 'all_flag' in kwargs.keys():
        all_flag = kwargs['all_flag']

    only_training = False
    if 'only_training' in kwargs.keys():
        if kwargs['only_training']:
            only_training = True

    plot_errors = False
    if 'plot_errors' in kwargs.keys():
        if kwargs['plot_errors']:
            plot_errors = True

    precannulation_values_per_bird = list()
    postcannulation_values_per_bird = list()
    pre_values_per_bird = list()
    fad_values_per_bird = list()
    post_values_per_bird = list()
    for i, file_name in enumerate(sorted(file_names)):
        if only_training:
            if 'SRCF002' in file_name:  # skip the repeated subject SRCF002 == SROF013
                continue
        cur_df = pd.read_excel(file_name)

        precannulation_data = cur_df[cur_df['Treatment'] == 'PRECANNULATION']
        postcannulation_data = cur_df[cur_df['Treatment'] == 'POSTCANNULATION']
        pre_data = cur_df[cur_df['Treatment'].str.match('^ACSF[1-9]+') == True]
        fad_data = cur_df[cur_df['Treatment'].str.match('^FAD[1-9]+') == True]
        post_data = cur_df[cur_df['Treatment'].str.match('^POSTACSF[1-9]+') == True]

        if column_to_plot not in ('hit_rate', 'rejection_rate'):
            precannulation_values_per_bird.append(list(precannulation_data[column_to_plot]))
            postcannulation_values_per_bird.append(list(postcannulation_data[column_to_plot]))
            pre_values_per_bird.append(list(pre_data[column_to_plot]))
            fad_values_per_bird.append(list(fad_data[column_to_plot]))
            post_values_per_bird.append(list(post_data[column_to_plot]))
        elif column_to_plot is 'hit_rate':
            precannulation_values_per_bird.append([hit / (hit + miss)*100 for
                                             hit, miss in zip(precannulation_data['hit'], precannulation_data['miss'])])
            postcannulation_values_per_bird.append([hit / (hit + miss)*100 for
                                             hit, miss in zip(postcannulation_data['hit'], postcannulation_data['miss'])])
            pre_values_per_bird.append([hit / (hit + miss)*100 for
                                        hit, miss in zip(pre_data['hit'], pre_data['miss'])])
            fad_values_per_bird.append([hit / (hit + miss)*100 for
                                        hit, miss in zip(fad_data['hit'], fad_data['miss'])])
            post_values_per_bird.append([hit / (hit + miss)*100 for
                                         hit, miss in zip(post_data['hit'], post_data['miss'])])
        elif column_to_plot is 'rejection_rate':
            precannulation_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(precannulation_data['reject'], precannulation_data['false_alarm'])])
            postcannulation_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(postcannulation_data['reject'], postcannulation_data['false_alarm'])])
            pre_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(pre_data['reject'], pre_data['false_alarm'])])
            fad_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(fad_data['reject'], fad_data['false_alarm'])])
            post_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(post_data['reject'], post_data['false_alarm'])])
        # x_axis = np.arange(1, 15)  # plot 2 weeks

    individual_plot_line_style = '-o'
    individual_plot_cmap_multiplier = 50
    individual_plot_color_alpha = 0.5

    mean_line_color = 'red'
    mean_line_style = '--'
    distance_between_plots = 0  # currently set at the end of loop to get the distance based on previous treatment
    for x_start_multiplier, cur_birds_data in \
            enumerate([precannulation_values_per_bird, postcannulation_values_per_bird]):
        cur_x_start = distance_between_plots * x_start_multiplier + 1
        max_days = np.max([len(animal_data) for animal_data in cur_birds_data])

        if column_to_plot not in ('hit_rate', 'rejection_rate'):
            if column_to_plot is not 'trials':
                ax.axhline(50, linestyle=':', color='black')
                ax.axhline(70, linestyle=':', color='goldenrod')

        for dummy_idx, bird_data in enumerate(cur_birds_data):
            ax.plot(np.arange(cur_x_start, cur_x_start + len(bird_data)),
                    bird_data, individual_plot_line_style,
                    color=plt.cm.jet(dummy_idx * individual_plot_cmap_multiplier), alpha=individual_plot_color_alpha)

        # for animal_data in cur_birds_data:
        #     while len(animal_data) < max_days:
        #         animal_data.append(np.NaN)
        #
        # means_to_plot = np.nanmean(cur_birds_data, axis=0)
        # error_to_plot = list()
        # for day in range(0, len(means_to_plot)):
        #     cur_day = np.array([animal_data[day] for animal_data in cur_birds_data])
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
        #     ax.plot(np.arange(cur_x_start, cur_x_start + len(cur_birds_data[0])),
        #             means_to_plot, linestyle=mean_line_style, color=mean_line_color, zorder=100, alpha=1)
        # else:
        #     ax.errorbar(np.arange(cur_x_start, cur_x_start + len(cur_birds_data[0])),
        #                 means_to_plot, linestyle=mean_line_style, yerr=error_to_plot, color=mean_line_color, zorder=100, alpha=1)
        ax.set_xticklabels([])
        ax.set_xticks([])
        distance_between_plots = max([len(each) for each in cur_birds_data]) + 2


    # ax.spines['bottom'].set_visible(False)

    if 'ylim' in kwargs.keys():
        ax.set_ylim(kwargs['ylim'])
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'])
    ax.set_xlabel('Days of training')

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

    path = ".\\Learning proof data"

    file_names = glob(path + r"\*xlsx")

    files = [file_name for file_name in file_names if 'SRCF' in file_name]

    pdf_name = 'SRCF_daily_cannulation.pdf'
    plot_errors = False

    with PdfPages(pdf_name) as pdf:
        plot_daily_treatment(files, '%correct', ylim=[0, 100], ylabel='% Correct', plot_errors=plot_errors)
        pdf.savefig()
        plt.close()

        plot_daily_treatment(files, 'hit_rate', ylim=[0, 100], ylabel='Hit rate', plot_errors=plot_errors)
        pdf.savefig()
        plt.close()

        plot_daily_treatment(files, 'rejection_rate', ylim=[0, 100], ylabel='Rejection rate', plot_errors=plot_errors)
        pdf.savefig()
        plt.close()

        plot_daily_treatment(files, 'd\'_loglinear_corrected', ylim=[-2.5, 2.5], ylabel='d\'', plot_errors=plot_errors)
        pdf.savefig()
        plt.close()

        plot_daily_treatment(files, 'trials', ylabel='Number of activations', plot_errors=plot_errors)
        pdf.savefig()
        plt.close()

    exit()