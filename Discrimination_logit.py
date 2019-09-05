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


def plot_logit(concatenated, title, colors_dict,
               csv_name='Logit_output.csv'):

    # rule_list = list(concatenated['Rule'])
    # rules = sorted(set(rule_list), key=rule_list.index)

    # days_list = list(concatenated['Day_of_training'])
    # days_of_training = sorted(set(days_list), key=days_list.index)

    treatment_list = list(concatenated['Treatment'])
    treatments = sorted(set(treatment_list), key=treatment_list.index)
    # exclude treatments not in colors_dict
    treatments = [treatment for treatment in treatments if treatment in colors_dict.keys()]

    datapoints_plot_treatment_alignment = np.linspace(0.1*len(treatments), -0.1*len(treatments), len(treatments))

    plt.clf()
    plt.cla()
    f = plt.figure()
    cax_list = list()
    ax = f.add_subplot(111)

    subsets = list()
    trials_per_treatment = list()
    for idx, treatment in enumerate(treatments):
        # # Restrict to first 5 days of training in each treatment
        # day_restriction_set = sorted(list(set(concatenated[concatenated['Treatment'] == treatment]['Day_of_training'])))[:5]
        # subset = concatenated[concatenated['Day_of_training'].isin(day_restriction_set)]
        cur_subset = concatenated[concatenated['Treatment'] == treatment]
        subsets.append(cur_subset)
        trials_per_treatment.append(len(cur_subset))

    trial_cap = min(trials_per_treatment)

    for idx, (subset, treatment) in enumerate(zip(subsets, treatments)):
        # Cap at the minimum number of trials among treatments
        capped_subset = subset[subset['Trial_number'] <= trial_cap]
        y = np.array(capped_subset['Score_per_trial'])
        X = np.arange(1, len(y) + 1)
        # Rescale X

        X = np.array([value/len(y)*100 for value in X])

        X = X[:, np.newaxis]

        # Fit the classifier
        clf = linear_model.LogisticRegression(solver='lbfgs')
        clf.fit(X, y)
        clf.score(X, y)

        jitter = np.random.normal(loc=0.0, scale=0.01, size=len(y))

        jittered_y = np.sum([y, jitter], axis=0)

        # and plot the result
        try:
            cax_list.append(ax.plot(X.ravel(), jittered_y + datapoints_plot_treatment_alignment[idx], marker='o',
                                    linestyle='', color=colors_dict[treatment], alpha=0.2))
        except IndexError:
            print('here')

        ax.axhline(y=0.5, linestyle=':', color='red')

        X_test = np.linspace(1, np.max(X) + 1, 300)

        loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
        ax.plot(X_test, loss, color=colors_dict[treatment])

        auc = integrate.trapz(loss, X_test)

        ax.set_ylabel('Probablity of Correct Response (Hits/Rejections)')
        ax.set_xlabel('% Trials (normalized)')

        # Add to CSV
        with open(str(csv_name), 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([title] + [treatment] + [float(clf.coef_)] + [float(clf.intercept_)] +[auc])

        # plt.tight_layout()

    legend_list = list()
    for treatment in treatments:
        legend_list.append(Line2D([], [], color=colors_dict[treatment], alpha=0.8, linestyle='-', label=treatment))

    labels = [h.get_label() for h in legend_list]

    f.legend(handles=legend_list, labels=labels, frameon=False, numpoints=1)

    ax.set_title(title)

    format_ax(ax)

    return f


def plot_average_logit(df, title, colors_dict={'PRE': 'royalblue', 'FAD': 'darkorange', 'POST': 'forestgreen'}):
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
                cur_curves.append(model(x_plot * float(cur_df['Logit_Coeff']) + float(cur_df['Logit_Intercept'])).ravel())
            except TypeError:
                continue

        for curve in cur_curves:
            ax.plot(x_plot, curve, color=colors_dict[treatment], alpha=0.1)

        mean_curve = np.mean(cur_curves, axis=0)
        se_curve = np.std(cur_curves, axis=0) / np.sqrt(np.count_nonzero(cur_curves, axis=0))

        ax.plot(x_plot, mean_curve, color=colors_dict[treatment], linewidth=mpl.rcParams['axes.linewidth']*3, linestyle='--')
        ax.fill_between(x_plot, mean_curve-se_curve, mean_curve+se_curve, alpha=0.1, color=colors_dict[treatment])

    legend_list = list()
    for treatment in treatments:
        legend_list.append(Line2D([], [], color=colors_dict[treatment], linestyle='-', label=treatment))

    labels = [h.get_label() for h in legend_list]

    # f.legend(handles=legend_list, labels=labels, frameon=False, numpoints=1)

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
    #

    path = getcwd()

    directories = [path + r'\SRCF discrimination logit data', path + r'\SROF discrimination logit data']
    # directories = [path + r'\Discrimination logit data\SRCF data']

    subj_dict = {}
    for project_folder in directories:
        if 'SRCF' in project_folder:
            treatments_and_colors_to_plot_list = [{'ACSF': 'royalblue',
                                                   'FAD': 'darkorange'},

                                                  {'ACSF': 'royalblue',
                                                   'BM100nL': 'lightcoral',
                                                   'BM200nL': 'firebrick',
                                                   'BM500nL': 'darkred'}]

        else:
            treatments_and_colors_to_plot_list = [{'ACSF': 'royalblue',
                                                   'FAD': 'darkorange'}]

        for treatments_and_colors_to_plot in treatments_and_colors_to_plot_list:
            if len(treatments_and_colors_to_plot.keys()) > 2:
                post_name = '_PREvsBACMUSC_discrimination_logit_regression'
            else:
                post_name = '_PREvsFAD_discrimination_logit_regression'

            file_name_pdf = project_folder + post_name
            csv_name = file_name_pdf + ".csv"
            with open(csv_name, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['Subject'] + ['Treatment'] + ['Logit_Coeff'] + ['Logit_Intercept'] + ['Area_under_curve'])

            with PdfPages(file_name_pdf + '.pdf') as pdf:

                for folder in [project_folder + "\\" + folder for folder in listdir(project_folder) if isdir(project_folder + "\\" + folder)]:
                    concatenated, file_names = load_data(folder, treatment_list=('ACSF', 'ACSF', 'FAD', 'FAD', 'BM100nL', 'BM200nL', 'BM500nL'), discrimination_data=True)
                    if '_GO' in file_name_pdf:
                        concatenated = concatenated[concatenated['Trial_type'] == 'GO']
                    elif '_NOGO' in file_name_pdf:
                        concatenated = concatenated[concatenated['Trial_type'] == 'NOGO']
                    concatenated = convert_to_score(concatenated, 1, 1, 0, 0)
                    subject_name = split("\\\\", folder)[-1]

                    f = plot_logit(concatenated, subject_name, csv_name=csv_name,
                                   colors_dict=treatments_and_colors_to_plot)

                    pdf.savefig()
                    plt.close(f)

            with PdfPages(file_name_pdf + '_discrimination_group_logits.pdf') as pdf:

                logit_df = pd.read_csv(csv_name)

                f = plot_average_logit(logit_df, split("\\\\", project_folder)[-1],
                                       colors_dict=treatments_and_colors_to_plot)

                pdf.savefig()
                plt.close(f)
    # exit()
