import numpy as np
from . import time_series
import pandas as pd
from datetime import datetime
from collections import OrderedDict


def split_by_category(input_data_frame, column_name):
    """
    Splits a time series data frame into a dictionary of separate time series: one for each level of a categorical variable.
    Each key is a category name and value is pandas dataframe of time series.

    :param input_data_frame: pandas data frame with timestamped views split by a categorical variable.
    :param column_name: (string) name of column containing the categorical variable.
    """

    # get levels of the category
    category_list = np.unique(input_data_frame[column_name].get_values())

    # separate categorised_time_series by category
    categorised_time_series = {}

    # extracts the initial time for the analysis and fills the categories with zeros
    # (some categories do not have views in all minutes)

    t0 = extract_date_time(input_data_frame.loc[0])

    input_data_frame = insert_date_time_columns(input_data_frame, t0)

    total_time_array = np.arange(0, input_data_frame.loc[len(input_data_frame)-1]['time_until_start']+1)

    for category in category_list:
        category_series = input_data_frame[input_data_frame[column_name] == category]
        category_series = fill_the_time_series_gaps(input_data_frame=category_series,
                                                    total_time_array=total_time_array)

        categorised_time_series[category] = category_series

    return categorised_time_series


def traffic_change(time_until_start, input_series, one_season, period=1440):
    """
    Calculates the DIFFERENCE, in number of views,
    of traffic change with respect to the prior average of a time series,
    from time_until_start, using input series and the average value at the time

    :param time_until_start: 
    :param input_series: 
    :param one_season: np array of length period with average for one season
    :param period: 
    :param window: 
    :return: 
    """

    # Compares with the same period of the day.

    time_in_the_period = time_until_start % period

    traffic_change_difference = input_series.loc[time_until_start] - one_season[time_in_the_period]
    return traffic_change_difference


def get_one_season_average(input_series, period, smooth=True, window=20):

    one_season = np.array([np.median(input_series[i::period]) for i in range(period)])

    if smooth:
        # Transforms into a series, performs a rolling average, fills the missing values and goes back
        # to numpy array
        # fills not a number with the first value

        one_season = pd.Series(one_season)
        one_season = one_season.rolling(window=window).mean()
        one_season = one_season.fillna(value=one_season[20])
        one_season = one_season.values

    return one_season



### PER-CATEGORY ANOMALIES

def get_categorised_anomalies(categorised_time_series, input_granularity='minute', detection_granularity='day',
                              detection_period=4):
    """
    Runs anomaly detection on the time series for each category and returns a dictionary of anomalies
    with keys as category names and values as dataframe of anomalies 

    :param categorised_time_series: dictionary of time series for each level of a category, e.g. section.
    :param categorised_time_series: 
    :param input_granularity: 
    :param detection_granularity: 
    :param detection_period: 
    :return: 
    """

    categorised_anomalies = {}

    for keys, values in categorised_time_series.items():
        print('%s '%keys, end = '')
        categorised_anomalies[keys] = time_series.predict_last_period(input_data_frame=values,
                                                                      input_granularity=input_granularity,
                                                                      detection_granularity=detection_granularity,
                                                                      detection_period=detection_period)
        # Add timestamps for anomaly end times
    print('\n')

    return categorised_anomalies


def list_of_explainers(global_anomalies, categorised_anomalies):
    """
    Identifies explainers by checking whether anomalies detected in the global time series also appear in time series 
    split by category (section, referrer, product, device, location).
    Returns a list of possible explainers for each global anomaly and saves this list in a new column called 'Explainers'
    in the global_anomalies dataframe.

    :param global_anomalies: pandas dataframe of global anomalies with start and end times.
    :param categorised_anomalies: pandas dataframe of anomalies for each level of a category, e.g. section (with start and end times).
    """

    explainers = []

    for index, row in global_anomalies.iterrows():

        list_of_explainers = []

        for key, value in categorised_anomalies.items():
            ser = categorised_anomalies[key]

            for index2, row2 in ser.iterrows():
                anomaly_overlap = overlap(start_time_global=row.date_time,
                                          end_time_global=row.end_date_time,
                                          start_time_categorised=row2.date_time,
                                          end_time_categorised=row2.end_date_time)
                if anomaly_overlap:
                    break

            if anomaly_overlap is True and len(categorised_anomalies[key] > 0):
                list_of_explainers.append(key)

        explainers.append(list_of_explainers)

    global_anomalies['explainer'] = explainers

    return global_anomalies



#######################################################################
# Convenience functions to add timestamp columns to pandas dataframes,#
# calculate overlaps in time series, etc                              #
#######################################################################


def overlap(start_time_global, end_time_global, start_time_categorised, end_time_categorised):
    """ Check for overlap between timestamps of global and categorised anomalies """
    if (start_time_categorised >= start_time_global and start_time_categorised < end_time_global) or \
            (start_time_global >= start_time_categorised and end_time_categorised > start_time_global):
        return True
    else:
        return False


def extract_date_time(d0):
    """ 
    Concatenates date, hour and minute into a timestamp.
    d0 = dataframe of anomalies containing columns for date, hour, minute and width.
    """
    return datetime.strptime(d0['date']+' '+str(d0['hour'])+' '+str(d0['minute']), '%Y-%m-%d %H %M')


def insert_date_time_columns(input_data_frame, t0):
    """
    Inserts columns with date_time stamp and time_until_start. 
    Assumes that time is in minutes.
    Compulsory parameter t0 is a datetime format that counts the time until start.

    :param input_data_frame: pandas dataframe with day, hour and minute columns.
    :return: input_data_frame with new column containing datetime rounded to minutes.

    """

    input_data_frame['date_time'] = 0
    input_data_frame['date_time'] = input_data_frame[['date', 'hour', 'minute']].apply(extract_date_time, axis=1)

    input_data_frame['time_until_start'] = (input_data_frame['date_time'] - t0).apply(
        lambda time_min: int(time_min.total_seconds() / 60))

    return input_data_frame


def fill_the_time_series_gaps(input_data_frame, total_time_array):
    """
    Fill the missing minutes in a time series and return a data frame with the missing zeros
    re-indexed according to time until start.
    
    The remanining fields of the dataframe are attributed 'dummy' values (0 or 'null' string)

    :param input_data_frame: 
    :param total_time_array: 
    :return: 
    """

    input_data_frame.set_index('time_until_start', drop=False, inplace=True)
    input_data_frame = input_data_frame.reindex(total_time_array, fill_value=0)

    return input_data_frame


def anomaly_decision_tree(percentage_changes_each_category, system_failure_tolerance=0.8):
    """
    
    :return: 
    """

    number_of_anomalous_categories = 0

    for p in percentage_changes_each_category:
        if abs(p) > system_failure_tolerance:
            number_of_anomalous_categories += 1

    if number_of_anomalous_categories > 0.7 * len(percentage_changes_each_category):
        return 'system_misbehavior'
    else:
        return 'social_engagement'


def flags(cats, average_t_ratio_anomaly, average_t_ratio_prior, detect):

    flag = OrderedDict()
    flag.keys = cats
    for cat in cats:
        ### if average_anomaly and average_prior are similar: white flag <<<<<< FINE TUNING
        if abs(average_t_ratio_anomaly[cat] - average_t_ratio_prior[cat]) <= 0.05:
            flag[cat] = 'white'
        if (average_t_ratio_anomaly[cat] - average_t_ratio_prior[cat]) > 0.05:
            flag[cat] = 'green'
        if (average_t_ratio_anomaly[cat] - average_t_ratio_prior[cat]) < -0.05:
            flag[cat] = 'red'
        print(cat, flag[cat])
        # print (abs(average_time_ratio_anomaly[cat] - average_time_ratio_prior[cat]))

    flag_system = 'System behaves normally'
    ### if all categories are fluctuating less than 0.05 but anomaly detected >>>>> system misbehaviour
    if detect:
        delta_list = (abs(average_t_ratio_anomaly[cat] - average_t_ratio_prior[cat]) for cat in cats)
        delta_max = max(delta_list)
        if delta_max < 0.06:  ### categories not responsible
            flag_system = 'System is misbehaving'
    print(flag_system)

    return[flag_system, flag]