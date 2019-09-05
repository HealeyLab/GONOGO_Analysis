__author__ = 'Matheus'
__notes__ = 'This code was recycled and adapted from an earlier bit of code with a different purpose. Will probably look ugly'
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


def pc_calculator(concatenated):
    hit_sum = np.sum(concatenated['Hit'])
    reject_sum = np.sum(concatenated['Reject'])

    return 100 * (hit_sum + reject_sum) / len(concatenated)


def dp_calculator(concatenated):
    hit_sum = np.sum(concatenated['Hit'])
    miss_sum = np.sum(concatenated['Miss'])
    reject_sum = np.sum(concatenated['Reject'])
    fa_sum = np.sum(concatenated['False_alarm'])
    return d_prime(hit_sum, miss_sum, reject_sum, fa_sum)


def get_hit_rate(concatenated):
    return np.sum(concatenated['Hit']) / (
        np.sum(concatenated['Miss']) + np.sum(concatenated['Hit']))


def get_cr_rate(concatenated):
    return np.sum(concatenated['Reject']) / (
        np.sum(concatenated['False_alarm']) + np.sum(concatenated['Reject']))


def get_miss_rate(concatenated):
    return np.sum(concatenated['Miss']) / (
        np.sum(concatenated['Miss']) + np.sum(concatenated['Hit']))


def get_fa_rate(concatenated):
    return np.sum(concatenated['False_alarm']) / (
        np.sum(concatenated['False_alarm']) + np.sum(concatenated['Reject']))


def model(x):
    return 1 / (1 + np.exp(-x))


def trial_initiation_rate(concatenated, title, csv_name):
    for idx, current_day in enumerate(sorted(list(set(concatenated['Day_of_training'])))):

        current_treatment = list(set(concatenated[concatenated['Day_of_training'] == current_day]['Treatment']))[0]

        current_day_df = concatenated[concatenated['Day_of_training'] == current_day]

        current_day_time_delta_min = (current_day_df['Time_from_start'].iloc[-1] - current_day_df['Time_from_start'].iloc[0]) / 60

        current_day_rate_min = len(current_day_df) / current_day_time_delta_min


        # ['Subject'] + ['Treatment'] +
        # ['Day1_%C'] + ['Day2_%C'] + ['Delta_%C'] +
        # ['Day1_d\''] + ['Day2_d\''] + ['Delta_d\''] +
        # ['Day1_GO_error'] + ['Day2_GO_error'] + ['Delta_GO_error'] +
        # ['Day1_NOGO_error'] + ['Day2_NOGO_error'] + ['Delta_NOGO_error']
        with open(str(csv_name), 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([title] + [current_day] + [current_treatment] + [len(current_day_df)] +
                            [current_day_time_delta_min] +
                            [current_day_rate_min] + [current_day_df['Time_from_start'].iloc[0]])


def plot_trial_per_day_data(df, column, title=''):
    treatment_list = list(df['Treatment'])
    treatments = sorted(set(treatment_list), key=treatment_list.index)

    plt.clf()
    plt.cla()
    f = plt.figure()
    cax_list = list()
    ax = f.add_subplot(111)

    # raw data first
    treatments_raw_data = list()
    for subject in set(df['Subject']):
        cur_subject_data = [
            np.mean(df[(df['Subject'] == subject) & (df['Treatment'] == 'Training')][column].values),
            np.mean(df[(df['Subject'] == subject) & (df['Treatment'] == 'PRE')][column].values),
            np.mean(df[(df['Subject'] == subject) & (df['Treatment'] == 'FAD')][column].values),
            np.mean(df[(df['Subject'] == subject) & (df['Treatment'] == 'POST')][column].values)
        ]

        treatments_raw_data.append(cur_subject_data)

        ax.plot([1, 2, 3, 4], cur_subject_data, 'o-', color='k', alpha=0.5)

    # Now group data
    cur_treatment_mean = np.nanmean(treatments_raw_data, axis=0)
    cur_treatment_se = np.nanstd(treatments_raw_data, axis=0) / np.sqrt(
        np.count_nonzero(~np.isnan(treatments_raw_data), axis=0) - 1)

    ax.errorbar([1, 2, 3, 4],
                cur_treatment_mean, color='r', yerr=cur_treatment_se, alpha=1,
                elinewidth=mpl.rcParams['lines.linewidth']*2,
                    linewidth=mpl.rcParams['lines.linewidth'] * 2,
                zorder=100)

    format_ax(ax, ylabel=column, xlabel='Treatment')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(('Training', 'PRE', 'FAD', 'POST'))
    ax.set_ylabel('Trials per day (average)')
    # f.suptitle(title)
    f.tight_layout()


if __name__ == "__main__":
    from os.path import isdir
    from os import chdir, listdir, getcwd
    from glob import glob
    from re import split

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
    # mpl.rcParams['lines.markersize'] = label_font_size / 8.
    path = getcwd()

    directories = [path + r'\SRCF data with training', path + r'\SROF data with training']
    # directories = [path + r'\SROF delta data']
    subj_dict = {}
    for project_folder in directories:
        file_name_pdf = project_folder + '_trials_per_minute'
        csv_name = file_name_pdf + ".csv"
        # with open(csv_name, 'w', newline='') as file:
        #     writer = csv.writer(file, delimiter=',')
        #     writer.writerow(['Subject'] + ['Day'] + ['Treatment'] + ['Trials'] + ['Time_delta_minutes'] +
        #                     ['Trials_per_minute'] +
        #                     ['First_trial_latency'])
        #
        # for folder in [project_folder + "\\" + folder for folder in listdir(project_folder) if isdir(project_folder + "\\" + folder)]:
        #     concatenated, file_names = load_data(folder, treatment_list=('Training', 'PRE', 'FAD', 'POST'))
        #     subject_name = split("\\\\", folder)[-1]
        #
        #     trial_initiation_rate(concatenated, subject_name, csv_name)

        with PdfPages(file_name_pdf + '_group_plots.pdf') as pdf:
            logit_df = pd.read_csv(csv_name)
            plot_trial_per_day_data(logit_df, 'Trials')
            pdf.savefig()
            plt.close()

        #     plot_group_data(logit_df, 'Delta_hit_rate')
        #     pdf.savefig()
        #     plt.close()
        #
        #     plot_group_data(logit_df, 'Delta_cr_rate')
        #     pdf.savefig()
        #     plt.close()
    exit()
