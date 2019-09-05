import numpy as np

from glob import glob
import pandas as pd
from scipy.stats import norm
from re import split
Z = norm.ppf


def d_prime(hits, misses, correct_rejections, false_alarms):
    # Loglinear unbiased correction for extreme cases, according to Hautus, 1995
    # Add 0.5 to hits and false alarms, and 1 to total signal and no-signal trials
    hit_rate = (hits+0.5)/(hits+misses+1)
    fa_rate = (false_alarms+0.5)/(false_alarms+correct_rejections+1)

    # Calculate d'
    d = Z(hit_rate) - Z(fa_rate)

    return d




def load_data(dir, treatment_list, discrimination_data=False):
    file_names = glob(dir + "\*go_nogo.csv")
    file_names.sort()  # the date format makes them sortable alphabetically

    # Batch process files
    df_list = list()
    day_of_training = 1
    for file_name in file_names:
        try:
            current_file = pd.read_csv(file_name, header=None, skiprows=1)
        except pd.errors.EmptyDataError:
            print('corrupted file!!! ' + file_name)
            continue

        # current_rule = file_name[-34:-19]
        split_current_file = split("_*_", split("\\\\", file_name)[-1])
        current_rule = "_".join([split_current_file[2], split_current_file[3]])
        current_file[8] = current_rule

        # if len(df_list) != 0:
        #     if df_list[-1][8].tail(1).item() != current_rule:  # If list isn't empty and last rule is different than current
        #         day_of_training = 1  # reset day of training

        current_file[9] = day_of_training
        day_of_training += 1
        df_list.append(current_file)

    concatenated = pd.concat(df_list, ignore_index=True)
    # concatenated.to_csv("test.csv")

    concatenated.columns = ['Trial_number', 'Trial_type', 'Response_time_s', 'Hit', 'Miss', 'Reject', 'False_alarm',
                            'Time_from_start', 'Rule', 'Day_of_training']

    concatenated['d_prime_partial'] = 0
    concatenated['%Correct_partial'] = 0
    concatenated['Treatment'] = ''
    concatenated['Day_of_treatment'] = 0
    concatenated['Hit_cumsum'] = 0
    concatenated['Miss_cumsum'] = 0
    concatenated['CR_cumsum'] = 0
    concatenated['FA_cumsum'] = 0
    rule_list = list(concatenated['Rule'])
    ordered_rules = sorted(set(rule_list), key=rule_list.index)
    if not discrimination_data:
        for unique_rule, treatment in zip(ordered_rules, treatment_list):
            subset = concatenated[concatenated['Rule'] == unique_rule]

            subset['Treatment'] = treatment
            subset['Trial_counter'] = np.arange(1, len(subset)+1)
            subset['Hit_cumsum'] = subset.Hit.cumsum()
            subset['Miss_cumsum'] = subset.Miss.cumsum()
            subset['CR_cumsum'] = subset.Reject.cumsum()
            subset['FA_cumsum'] = subset['False_alarm'].cumsum()

            subset['d_prime_partial'] = d_prime(subset['Hit_cumsum'], subset['Miss_cumsum'],
                                                      subset['CR_cumsum'], subset['FA_cumsum'])

            subset['%Correct_partial'] = (subset['Hit_cumsum'] + subset['CR_cumsum']) / \
                                               subset['Trial_counter'] * 100

            # fill the main sheet
            concatenated['Hit_cumsum'][concatenated['Rule'] == unique_rule] = subset['Hit_cumsum']
            concatenated['Miss_cumsum'][concatenated['Rule'] == unique_rule] = subset['Miss_cumsum']
            concatenated['CR_cumsum'][concatenated['Rule'] == unique_rule] = subset['CR_cumsum']
            concatenated['FA_cumsum'][concatenated['Rule'] == unique_rule] = subset['FA_cumsum']

            concatenated['Trial_number'][concatenated['Rule'] == unique_rule] = subset['Trial_counter']
            concatenated['d_prime_partial'][concatenated['Rule'] == unique_rule] = subset['d_prime_partial']
            concatenated['%Correct_partial'][concatenated['Rule'] == unique_rule] = subset['%Correct_partial']
            concatenated['Treatment'][concatenated['Rule'] == unique_rule] = subset['Treatment']

    else:
        day_list = list(concatenated['Day_of_training'])
        ordered_days = sorted(set(day_list), key=day_list.index)
        for dummy_idx, day in enumerate(ordered_days):
            subset = concatenated[concatenated['Day_of_training'] == day]
            subset['Trial_counter'] = np.arange(1, len(subset) + 1)
            subset['Hit_cumsum'] = subset.Hit.cumsum()
            subset['Miss_cumsum'] = subset.Miss.cumsum()
            subset['CR_cumsum'] = subset.Reject.cumsum()
            subset['FA_cumsum'] = subset['False_alarm'].cumsum()

            subset['d_prime_partial'] = d_prime(subset['Hit_cumsum'], subset['Miss_cumsum'],
                                                subset['CR_cumsum'], subset['FA_cumsum'])

            subset['%Correct_partial'] = (subset['Hit_cumsum'] + subset['CR_cumsum']) / \
                                         subset['Trial_counter'] * 100

            subset['Treatment'] = treatment_list[dummy_idx]

            # fill the main sheet
            concatenated['Trial_number'][concatenated['Day_of_training'] == day] = subset['Trial_counter']
            concatenated['d_prime_partial'][concatenated['Day_of_training'] == day] = subset['d_prime_partial']
            concatenated['%Correct_partial'][concatenated['Day_of_training'] == day] = subset['%Correct_partial']

            if treatment_list is not None:
                concatenated['Treatment'][concatenated['Day_of_training'] == day] = subset['Treatment']

    return concatenated, file_names
