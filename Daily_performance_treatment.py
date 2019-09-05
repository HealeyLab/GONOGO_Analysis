__author__ = 'Matheus'
import matplotlib.pyplot as plt
import matplotlib as mpl
from re import search, split
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

    training_values_per_bird = list()
    pre_values_per_bird = list()
    fad_values_per_bird = list()
    post_values_per_bird = list()
    for i, file_name in enumerate(sorted(file_names)):
        if only_training:
            if 'SRCF002' in file_name:  # skip the repeated subject SRCF002 == SROF013
                continue
        cur_df = pd.read_excel(file_name)
        if 'SROF' in file_name:
            training_data = cur_df[pd.isna(cur_df['Treatment'])]
            pre_data = cur_df[cur_df['Treatment'].str.match('^SALINE[1-9]+') == True]
            fad_data = cur_df[cur_df['Treatment'].str.match('^FAD[1-9]+') == True]
            post_data = cur_df[cur_df['Treatment'].str.match('^POSTSALINE[1-9]+') == True]
        else:
            training_data = cur_df[cur_df['Treatment'] == 'POSTCANNULATION']
            pre_data = cur_df[cur_df['Treatment'].str.match('^ACSF[1-9]+') == True]
            fad_data = cur_df[cur_df['Treatment'].str.match('^FAD[1-9]+') == True]
            post_data = cur_df[cur_df['Treatment'].str.match('^POSTACSF[1-9]+') == True]

        if column_to_plot not in ('hit_rate', 'rejection_rate'):
            training_values_per_bird.append(list(training_data[column_to_plot]))
            pre_values_per_bird.append(list(pre_data[column_to_plot]))
            fad_values_per_bird.append(list(fad_data[column_to_plot]))
            post_values_per_bird.append(list(post_data[column_to_plot]))
        elif column_to_plot is 'hit_rate':
            training_values_per_bird.append([hit / (hit + miss)*100 for hit, miss in zip(training_data['hit'].values, training_data['miss'].values)])
            pre_values_per_bird.append([hit / (hit + miss)*100 for hit, miss in zip(pre_data['hit'], pre_data['miss'])])
            fad_values_per_bird.append([hit / (hit + miss)*100 for hit, miss in zip(fad_data['hit'], fad_data['miss'])])
            post_values_per_bird.append([hit / (hit + miss)*100 for hit, miss in zip(post_data['hit'], post_data['miss'])])
        elif column_to_plot is 'rejection_rate':
            training_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(training_data['reject'], training_data['false_alarm'])])
            pre_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(pre_data['reject'], pre_data['false_alarm'])])
            fad_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(fad_data['reject'], fad_data['false_alarm'])])
            post_values_per_bird.append([reject / (reject + false_alarm)*100 for
                                             reject, false_alarm in zip(post_data['reject'], post_data['false_alarm'])])

        if column_to_plot is 'trials':
            subject_name = split('_*_', split("\\\\", file_name)[-1])[0]
            with open(csv_name, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')

                for treatment, df in zip(['Training', 'PRE', 'FAD', 'POST'], [training_data, pre_data, fad_data, post_data]):
                    for day_of_treatment_idx in np.arange(0, len(df)):
                        writer.writerow([subject_name] + [treatment] +
                                        [day_of_treatment_idx + 1] + [df['trials'].iloc[day_of_treatment_idx]])

    individual_plot_line_style = '-o'
    individual_plot_cmap_multiplier = 50
    individual_plot_color_alpha = 0.5

    mean_line_color = 'red'
    mean_line_style = '--'
    distance_between_plots = 0  # currently set at the end of loop to get the distance based on previous treatment
    cur_x_start = 1
    if not only_training:
        lists_to_plot = [training_values_per_bird, pre_values_per_bird, fad_values_per_bird, post_values_per_bird]
    else:
        lists_to_plot = [training_values_per_bird]

    for x_start_multiplier, cur_birds_data in \
            enumerate(lists_to_plot):

        max_days = np.max([len(animal_data) for animal_data in cur_birds_data])
        for animal_data in cur_birds_data:
            while len(animal_data) < max_days:
                animal_data.append(np.NaN)

        if column_to_plot is not 'trials':
            ax.axhline(50, linestyle=':', color='black')
            if column_to_plot not in ('hit_rate', 'rejection_rate'):
                ax.axhline(70, linestyle=':', color='goldenrod')

        for dummy_idx, bird_data in enumerate(cur_birds_data):
            ax.plot(np.arange(cur_x_start, cur_x_start + len(bird_data)),
                    bird_data, individual_plot_line_style,
                    color=plt.cm.jet(dummy_idx * individual_plot_cmap_multiplier),
                    alpha=individual_plot_color_alpha)

        means_to_plot = np.nanmean(cur_birds_data, axis=0)

        error_to_plot = list()
        for day in range(0, len(means_to_plot)):
            cur_day = np.array([animal_data[day] for animal_data in cur_birds_data])
            cur_day = cur_day[~np.isnan(cur_day)]
            error_to_plot.append(np.nanstd(cur_day, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(cur_day), axis=0)))

            # cur_ci = stats.t.interval(0.95, len(cur_day) - 1, loc=np.mean(cur_day), scale=stats.sem(cur_day))
            # cis_to_plot.append(cur_ci[1] - cur_ci[0])

        # Plot means and CIs
        # ax.errorbar(np.arange(1, len(means_to_plot)+1), means_to_plot, yerr=error_to_plot, color='k', zorder=100)
        # ax.errorbar(np.arange(cur_x_start, cur_x_start + len(cur_birds_data[0])),
        #             means_to_plot, yerr=error_to_plot, color='r', zorder=100, alpha=0.5)
        # if not plot_errors:
        #     ax.plot(np.arange(cur_x_start, cur_x_start + len(cur_birds_data[0])),
        #             means_to_plot, linestyle=mean_line_style, color=mean_line_color, zorder=100, alpha=1)
        # else:
        #     ax.errorbar(np.arange(cur_x_start, cur_x_start + len(cur_birds_data[0])),
        #                 means_to_plot, linestyle=mean_line_style, yerr=error_to_plot, color=mean_line_color,
        #                 zorder=100, alpha=1)
        ax.set_xticklabels([])
        ax.set_xticks([])
        cur_x_start += max([len(each) for each in cur_birds_data]) + 5

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
    # mpl.rcParams['lines.markersize'] = label_font_size / 3.
    # mpl.rcParams['lines.linewidth'] = label_font_size / 6.

    path = ".\\Learning proof data"

    file_names = glob(path + r"\*xlsx")

    srof_file_names = [file_name for file_name in file_names if 'SROF' in file_name]
    srcf_file_names = [file_name for file_name in file_names if 'SRCF' in file_name]

    pdf_names = ['SROF_daily_treatment.pdf', 'SRCF_daily_treatment.pdf', 'ALL_daily_training.pdf']
    only_training_list = [False, False, True]
    plot_errors = False
    for pdf_name, files, only_training in zip(pdf_names, (srof_file_names, srcf_file_names, file_names), only_training_list):
        csv_name = pdf_name[:-4] + "_number_of_trials.csv"
        with open(csv_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Subject'] + ['Treatment'] + ['Day_of_treatment'] + ['Number_of_trials'])


        with PdfPages(pdf_name) as pdf:
            plot_daily_treatment(files, '%correct', ylim=[0, 100], ylabel='% Correct', only_training=only_training, plot_errors=plot_errors)
            pdf.savefig()
            plt.close()

            plot_daily_treatment(files, 'hit_rate', ylim=[0, 100], ylabel='Hit rate', only_training=only_training, plot_errors=plot_errors)
            pdf.savefig()
            plt.close()

            plot_daily_treatment(files, 'rejection_rate', ylim=[0, 100], ylabel='Rejection rate', only_training=only_training, plot_errors=plot_errors)
            pdf.savefig()
            plt.close()

            plot_daily_treatment(files, 'd\'_loglinear_corrected', ylim=[-2.5, 2.5], ylabel='d\'', only_training=only_training, plot_errors=plot_errors)
            pdf.savefig()
            plt.close()

            plot_daily_treatment(files, 'trials', ylabel='Number of activations', only_training=only_training, plot_errors=plot_errors)
            pdf.savefig()
            plt.close()
