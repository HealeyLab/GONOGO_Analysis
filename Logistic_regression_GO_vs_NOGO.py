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

    return 100 * (hit_sum + reject_sum) / len(concatenated['Hit'])


def score_calculator(concatenated, scores):
    temp_df = convert_to_score(concatenated, scores[0], scores[1], scores[2], scores[3])

    return np.sum((temp_df['Hit_score'], temp_df['CR_score'], temp_df['Miss_score'], temp_df['FA_score']))


def dp_calculator(concatenated):
    hit_sum = np.sum(concatenated['Hit'])
    miss_sum = np.sum(concatenated['Miss'])
    reject_sum = np.sum(concatenated['Reject'])
    fa_sum = np.sum(concatenated['False_alarm'])
    return d_prime(hit_sum, miss_sum, reject_sum, fa_sum)


def model(x):
    return 1 / (1 + np.exp(-x))


def plot_logit(concatenated, title, colors_dict, csv_name='Logit_output.csv'):

    # rule_list = list(concatenated['Rule'])
    # rules = sorted(set(rule_list), key=rule_list.index)

    # colors_dict = {'Training': 'rebeccapurple', 'PRE': 'royalblue', 'FAD': 'darkorange', 'POST': 'forestgreen'}

    # days_list = list(concatenated['Day_of_training'])
    # days_of_training = sorted(set(days_list), key=days_list.index)

    treatment_list = list(concatenated['Treatment'])
    treatments = sorted(set(treatment_list), key=treatment_list.index)

    datapoints_plot_treatment_alignment = [0.05, 0, -0.05]

    plt.clf()
    plt.cla()
    f = plt.figure()
    cax_list = list()
    ax = f.add_subplot(111)
    subsets = list()
    trials_per_treatment = list()
    for idx, treatment in enumerate(treatments):
        cur_subset = concatenated[concatenated['Treatment'] == treatment]

        subsets.append(cur_subset.reset_index())
        trials_per_treatment.append(len(cur_subset))

    trial_cap = min(trials_per_treatment)

    for idx, (subset, treatment) in enumerate(zip(subsets, treatments)):
        # Cap at the minimum number of trials among treatments
        capped_subset = subset.drop(subset.index[trial_cap:])
        y = np.array(capped_subset['Score_per_trial'])
        X = np.arange(1, len(y) + 1)
        # Rescale X

        X = np.array([value/len(y)*100 for value in X])

        X = X[:, np.newaxis]

        # Fit the classifier
        clf = linear_model.LogisticRegression(solver='lbfgs')
        try:
            clf.fit(X, y)
        except ValueError:  # When not enough data (hour restriction) it breaks
            continue
        clf.score(X, y)

        jitter = np.random.normal(loc=0.0, scale=0.01, size=len(y))

        jittered_y = np.sum([y, jitter], axis=0)

        # and plot the result
        cax_list.append(ax.plot(X.ravel(), jittered_y + datapoints_plot_treatment_alignment[idx], marker='o',
                                linestyle='', color=colors_dict[treatment], alpha=0.2))

        ax.axhline(y=0.5, linestyle=':', color='black')

        X_test = np.linspace(1, np.max(X) + 1, 300)

        loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
        ax.plot(X_test, loss, color=colors_dict[treatment])

        ax.set_ylabel('Probablity of Correct Response (Hits/Rejections)')
        ax.set_xlabel('% Trials (normalized)')

        auc = integrate.trapz(loss, X_test)

        percent_correct = 100 * (np.sum(capped_subset['Hit']) + np.sum(capped_subset['Reject'])) / len(capped_subset)

        go_error = 100 * (np.sum(capped_subset['Miss']) / (np.sum(capped_subset['Hit']) + np.sum(capped_subset['Miss'])))
        nogo_error = 100 * (np.sum(capped_subset['False_alarm']) / (np.sum(capped_subset['False_alarm']) + np.sum(capped_subset['Reject'])))

        # Add to CSV
        with open(str(csv_name), 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([title] + [treatment] + [float(clf.coef_)] +
                            [float(clf.intercept_)] + [auc] + [percent_correct] +
                            [go_error] + [nogo_error])

        plt.tight_layout()

    pre_legend = Line2D([], [], color=colors_dict['PRE'], alpha=0.8, linestyle='-', label='PRE')
    FAD_legend = Line2D([], [], color=colors_dict['FAD'], alpha=0.8, linestyle='-', label='FAD')
    post_legend = Line2D([], [], color=colors_dict['POST'], alpha=0.8, linestyle='-', label='POST')

    handles = [pre_legend, FAD_legend, post_legend]
    labels = [h.get_label() for h in handles]

    f.legend(handles=handles, labels=labels, frameon=False, numpoints=1)

    ax.set_title(title)

    format_ax(ax)

    return f


def plot_average_logit(df, title, colors_dict):
    treatment_list = list(df['Treatment'])
    treatments = sorted(set(treatment_list), key=treatment_list.index)

    plt.clf()
    plt.cla()
    f = plt.figure()
    cax_list = list()
    ax = f.add_subplot(111)
    ax.axhline(y=0.5, linestyle=':', color='black')
    for idx, treatment in enumerate(treatments):
        x_plot = np.linspace(1, 100, 300)

        cur_curves = list()
        for subject in sorted(list(set(df['Subject']))):
            cur_df = df[(df['Subject'] == subject) & (df['Treatment'] == treatment)]
            try:
                cur_curve = model(x_plot * float(cur_df['Logit_Coeff']) + float(cur_df['Logit_Intercept'])).ravel()
            except TypeError:  # SROF20 doesn't have POST yet
                continue
            ax.plot(x_plot, cur_curve, color=colors_dict[treatment], alpha=0.1)
            cur_curves.append(cur_curve)

        # for curve in cur_curves:
        #     ax.plot(x_plot, curve, color=colors_dict[treatment], alpha=0.2)

        mean_curve = np.mean(cur_curves, axis=0)
        se_curve = np.std(cur_curves, axis=0) / np.sqrt(np.count_nonzero(cur_curves, axis=0))

        ax.plot(x_plot, mean_curve, color=colors_dict[treatment], linewidth=mpl.rcParams['axes.linewidth']*3, linestyle='--')
        ax.fill_between(x_plot, mean_curve-se_curve, mean_curve+se_curve, alpha=0.1, color=colors_dict[treatment])

    pre_legend = Line2D([], [], color=colors_dict['PRE'], linestyle='-', label='PRE')
    FAD_legend = Line2D([], [], color=colors_dict['FAD'], linestyle='-', label='FAD')
    post_legend = Line2D([], [], color=colors_dict['POST'], linestyle='-', label='POST')

    handles = [pre_legend, FAD_legend, post_legend]
    labels = [h.get_label() for h in handles]

    # f.legend(handles=handles, labels=labels, frameon=False, numpoints=1)

    # ax.set_title(title)
    ax.set_ylim([0, 1])

    ax.set_ylabel('Probablity of Correct Response')
    ax.set_xlabel('% Trials (normalized)')

    format_ax(ax)
    plt.tight_layout()
    return f


def convert_to_score(concatenated, hit_score, cr_score, miss_score, fa_score):
    # Convert responses into scores
    hit_score = hit_score
    cr_score = cr_score
    miss_score = miss_score
    fa_score = fa_score

    ret_df = concatenated[:]
    ret_df['Hit_score'] = 0
    ret_df['CR_score'] = 0
    ret_df['Miss_score'] = 0
    ret_df['FA_score'] = 0
    ret_df['Score_partial'] = 0
    ret_df['Score_fraction'] = 0
    ret_df['Score_fraction_cumsum'] = 0
    ret_df['Score_per_trial'] = 0
    ret_df.loc[ret_df['Hit'] == 1, 'Hit_score'] = hit_score
    ret_df.loc[ret_df['Reject'] == 1, 'CR_score'] = cr_score
    ret_df.loc[ret_df['Miss'] == 1, 'Miss_score'] = miss_score
    ret_df.loc[ret_df['False_alarm'] == 1, 'FA_score'] = fa_score

    for unique_rule in set(ret_df['Rule']):
        subset = ret_df[ret_df['Rule'] == unique_rule]
        subset['Score_per_trial'] = subset.loc[:, ['Hit_score', 'CR_score', 'Miss_score', 'FA_score']].sum(axis=1)  # Get score of each trial to be cumsummed

        subset['Score_partial'] = subset['Score_per_trial'].cumsum()

        ret_df['Score_partial'][ret_df['Rule'] == unique_rule] = subset['Score_partial']
        ret_df['Score_per_trial'][ret_df['Rule'] == unique_rule] = subset['Score_per_trial']

    # After figuring out the total score, get the fractioned score
    for unique_rule in set(ret_df['Rule']):
        subset = ret_df[ret_df['Rule'] == unique_rule]
        score_row_sum = subset.loc[:, ['Hit_score', 'CR_score', 'Miss_score', 'FA_score']].sum(axis=1)  # Get score of each trial to be cumsummed
        max_score = np.max(subset['Score_partial'])

        subset['Score_fraction'] = score_row_sum / max_score
        subset['Score_fraction_cumsum'] = subset.Score_fraction.cumsum()

        ret_df['Score_fraction'][ret_df['Rule'] == unique_rule] = subset['Score_fraction']
        ret_df['Score_fraction_cumsum'][ret_df['Rule'] == unique_rule] = subset['Score_fraction_cumsum']

    return ret_df


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

    directories = [path + r'\SRCF logit data', path + r'\SROF logit data']
    # directories = [path + r'\SROF data']
    colors_dict = {'PRE': 'royalblue', 'FAD': 'darkorange', 'POST': 'forestgreen'}
    subj_dict = {}
    for project_folder in directories:
        file_name_pdf_list = [project_folder + '_logit_regression_unrestricted',
                              project_folder + '_logit_regression_unrestricted_GO',
                              project_folder + '_logit_regression_unrestricted_NOGO']
        # file_name_pdf_list = [project_folder + '_logit_regression_unrestricted_Bias']
        for file_name_pdf in file_name_pdf_list:
            csv_name = file_name_pdf + ".csv"
            with open(csv_name, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['Subject'] + ['Treatment'] + ['Logit_Coeff'] +
                                ['Logit_Intercept'] + ['Area_under_curve'] +
                                ['%Correct'] + ['Hit_rate'] + ['Rejection_rate'])

            with PdfPages(file_name_pdf + '.pdf') as pdf:
                for folder in [project_folder + "\\" + folder for folder in listdir(project_folder) if isdir(project_folder + "\\" + folder)]:
                    concatenated, file_names = load_data(folder, treatment_list=['PRE', 'FAD', 'POST'])
                    if '_GO' in file_name_pdf:
                        concatenated = concatenated[concatenated['Trial_type'] == 'GO']
                    elif '_NOGO' in file_name_pdf:
                        concatenated = concatenated[concatenated['Trial_type'] == 'NOGO']

                    if 'Bias' in file_name_pdf:
                        concatenated = convert_to_score(concatenated, 1, 0, 0, 1)
                    else:
                        concatenated = convert_to_score(concatenated, 1, 1, 0, 0)
                    subject_name = split("\\\\", folder)[-1]

                    f = plot_logit(concatenated, subject_name, colors_dict, csv_name)

                    pdf.savefig()
                    plt.close(f)

            with PdfPages(file_name_pdf + '_group_logits.pdf') as pdf:

                logit_df = pd.read_csv(csv_name)

                f = plot_average_logit(logit_df, split("\\\\", project_folder)[-1], colors_dict)

                pdf.savefig()
                plt.close(f)

