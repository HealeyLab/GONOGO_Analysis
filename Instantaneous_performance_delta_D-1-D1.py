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


def response_bias(hits, misses, correct_rejections, false_alarms):
    # Loglinear (Hautus, 1995) response bias (Macmillan & Creelman, 1991)
    # Add 0.5 to hits and false alarms, and 1 to total signal and no-signal trials
    # hit_rate = (hits+0.5)/(hits+misses+1)
    # fa_rate = (false_alarms+0.5)/(false_alarms+correct_rejections+1)

    hit_rate = (hits)/(hits+misses)
    fa_rate = (false_alarms)/(false_alarms+correct_rejections)
    # Calculate d'
    rb = -0.5 * (Z(hit_rate) + Z(fa_rate))

    return rb


def response_bias_calculator(concatenated):
    hit_sum = np.sum(concatenated['Hit'])
    miss_sum = np.sum(concatenated['Miss'])
    reject_sum = np.sum(concatenated['Reject'])
    fa_sum = np.sum(concatenated['False_alarm'])
    return response_bias(hit_sum, miss_sum, reject_sum, fa_sum)


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


def inst_performance_delta(concatenated, title, csv_name, inst_trial_n=5):

    treatment_list = list(concatenated['Treatment'])
    treatments = sorted(set(treatment_list), key=treatment_list.index)

    previous_last_day = 0
    for idx, treatment in enumerate(treatments):
        if previous_last_day > 0:
            first_day = sorted(list(set(concatenated[concatenated['Treatment'] == treatment]['Day_of_training'])))[0]

            day1_last_trials = concatenated[concatenated['Day_of_training'] == previous_last_day].iloc[-inst_trial_n:]
            day2_first_trials = concatenated[concatenated['Day_of_training'] == first_day].iloc[:inst_trial_n]

            day1_last_GOs = concatenated[(concatenated['Trial_type'] == 'GO') &
                                         (concatenated['Day_of_training'] == previous_last_day)].iloc[-inst_trial_n:]
            day2_first_GOs = concatenated[(concatenated['Trial_type'] == 'GO') &
                                         (concatenated['Day_of_training'] == first_day)].iloc[:inst_trial_n]

            day1_last_NOGOs = concatenated[(concatenated['Trial_type'] == 'NOGO') &
                                         (concatenated['Day_of_training'] == previous_last_day)].iloc[-inst_trial_n:]
            day2_first_NOGOs = concatenated[(concatenated['Trial_type'] == 'NOGO') &
                                         (concatenated['Day_of_training'] == first_day)].iloc[:inst_trial_n]

            pc_day1 = pc_calculator(day1_last_trials)
            pc_day2 = pc_calculator(day2_first_trials)

            dp_day1 = dp_calculator(day1_last_trials)
            dp_day2 = dp_calculator(day2_first_trials)

            day1_hit_rate = get_hit_rate(day1_last_GOs)
            day2_hit_rate = get_hit_rate(day2_first_GOs)

            day1_miss_rate = get_miss_rate(day1_last_GOs)
            day2_miss_rate = get_miss_rate(day2_first_GOs)

            day1_fa_rate = get_fa_rate(day1_last_NOGOs)
            day2_fa_rate = get_fa_rate(day2_first_NOGOs)

            day1_cr_rate = get_cr_rate(day1_last_NOGOs)
            day2_cr_rate = get_cr_rate(day2_first_NOGOs)

            day1_rb = response_bias_calculator(day1_last_trials)
            day2_rb = response_bias_calculator(day2_first_trials)

            # ['Subject'] + ['Treatment'] +
            # ['Day1_%C'] + ['Day2_%C'] + ['Delta_%C'] +
            # ['Day1_d\''] + ['Day2_d\''] + ['Delta_d\''] +
            # ['Day1_GO_error'] + ['Day2_GO_error'] + ['Delta_GO_error'] +
            # ['Day1_NOGO_error'] + ['Day2_NOGO_error'] + ['Delta_NOGO_error'] +
            # ['Day-1_response_bias'] + ['Day1_response_bias'] + ['Delta_response_bias']
            with open(str(csv_name), 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow([title] + [treatment] +
                                [pc_day1] + [pc_day2] + [pc_day2 - pc_day1] +
                                [dp_day1] + [dp_day2] + [dp_day2 - dp_day1] +
                                [day1_hit_rate] + [day2_hit_rate] + [day2_hit_rate - day1_hit_rate] +
                                [day1_miss_rate] + [day2_miss_rate] + [day2_miss_rate - day1_miss_rate] +
                                [day1_fa_rate] + [day2_fa_rate] + [day2_fa_rate - day1_fa_rate] +
                                [day1_cr_rate] + [day2_cr_rate] + [day2_cr_rate - day1_cr_rate] +
                                [day1_rb] + [day2_rb] + [day2_rb - day1_rb])

        previous_last_day = sorted(list(set(concatenated[concatenated['Treatment'] == treatment]['Day_of_training'])))[
            -1]


def plot_group_data(df, title, pdf_handle):

    treatment_list = list(df['Treatment'])
    treatments = sorted(set(treatment_list), key=treatment_list.index)

    y_labels = [r'$\Delta$ % Correct', r'$\Delta$ d' + '\'', r'$\Delta$ Hit rate',
                r'$\Delta$ Rejection rate', 'Day 1 Hit Rate', 'Day 1 Rejection Rate',
                'Day 1 response bias', r'$\Delta$ Response bias']

    for dummy_idx, column in \
            enumerate(['Delta_%C', 'Delta_d\'', 'Delta_hit_rate',
                       'Delta_cr_rate', 'Day1_hit_rate', 'Day1_cr_rate',
                       'Day1_response_bias', 'Delta_response_bias']):
        plt.clf()
        plt.cla()
        f = plt.figure()
        cax_list = list()
        ax = f.add_subplot(111)

        # raw data first
        treatments_raw_data = list()
        for subject in set(df['Subject']):
            if column in ['Delta_%C', 'Delta_d\'']:
                cur_subject_data = [
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'PRE')][column].values),
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'FAD')][column].values),
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'POST')][column].values)
                ]

            elif 'response_bias' in column:
                cur_subject_data = [
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'PRE')][column].values),
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'FAD')][column].values),
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'POST')][column].values)
                ]
            else:
                cur_subject_data = [
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'PRE')][column].values)*100,
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'FAD')][column].values)*100,
                    float(df[(df['Subject'] == subject) & (df['Treatment'] == 'POST')][column].values)*100
                ]

            treatments_raw_data.append(cur_subject_data)

            ax.plot([1, 2, 3], cur_subject_data, 'o-', color='k', alpha=0.5)

        # Now group data
        cur_treatment_mean = np.nanmean(treatments_raw_data, axis=0)
        cur_treatment_se = np.nanstd(treatments_raw_data, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(treatments_raw_data), axis=0) - 1)

        ax.errorbar([1, 2, 3],
               cur_treatment_mean, color='r', yerr=cur_treatment_se, alpha=1, elinewidth=mpl.rcParams['lines.linewidth']*2,
                    linewidth=mpl.rcParams['lines.linewidth'] * 2,
                    zorder=100)

        format_ax(ax, ylabel=column, xlabel='Treatment')

        ax.axhline(y=0, linestyle=':', color='blue')

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(('PRE', 'FAD', 'POST'))
        ax.set_ylabel(y_labels[dummy_idx])
        if column in ['Delta_%C', 'Delta_hit_rate', 'Delta_cr_rate']:
            ax.set_ylim([-80, 20])
        # f.suptitle(title)
        f.tight_layout()
        pdf_handle.savefig()
        plt.close(f)


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

    directories = [path + r'\SRCF delta data', path + r'\SROF delta data']
    # directories = [path + r'\SROF delta data']
    inst_trial_n = 50
    subj_dict = {}
    for project_folder in directories:
        file_name_pdf = project_folder + '_inst_' + str(inst_trial_n) + 'trials_delta_D-1_vs_D1'
        csv_name = file_name_pdf + ".csv"
        with open(csv_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Subject'] + ['Treatment'] +
                            ['Day-1_%C'] + ['Day1_%C'] + ['Delta_%C'] +
                            ['Day-1_d\''] + ['Day1_d\''] + ['Delta_d\''] +
                            ['Day-1_hit_rate'] + ['Day1_hit_rate'] + ['Delta_hit_rate'] +
                            ['Day-1_miss_rate'] + ['Day1_miss_rate'] + ['Delta_miss_rate'] +
                            ['Day-1_fa_rate'] + ['Day1_fa_rate'] + ['Delta_fa_rate'] +
                            ['Day-1_cr_rate'] + ['Day1_cr_rate'] + ['Delta_cr_rate'] +
                            ['Day-1_response_bias'] + ['Day1_response_bias'] + ['Delta_response_bias'])

        for folder in [project_folder + "\\" + folder for folder in listdir(project_folder) if isdir(project_folder + "\\" + folder)]:
            concatenated, file_names = load_data(folder, treatment_list=('Training', 'PRE', 'FAD', 'POST'))
            subject_name = split("\\\\", folder)[-1]

            inst_performance_delta(concatenated, subject_name, csv_name, inst_trial_n=inst_trial_n)

        with PdfPages(file_name_pdf + '_group_plots.pdf') as pdf:
            logit_df = pd.read_csv(csv_name)

            f = plot_group_data(logit_df, split("\\\\", project_folder)[-1], pdf)
    # exit()
