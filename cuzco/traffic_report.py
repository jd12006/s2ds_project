import numpy as np
import pandas as pd
import json
import sys

from .anomaly_explanation import get_one_season_average
from .anomaly_explanation import traffic_change
from .anomaly_explanation import anomaly_decision_tree
from .anomaly_explanation import split_by_category
from .anomaly_explanation import get_categorised_anomalies
from .anomaly_explanation import list_of_explainers

from .get_views import get_views_column
from .get_views import get_views_simple

from .time_series import predict_last_period

import matplotlib.pyplot as plt
import seaborn as sns
import time

from datetime import datetime, timedelta


def traffic_report_txt(list_of_data_frames, overall_data_frame, list_of_categories, anomaly_description,
                       one_day=1440, system_failure_tolerance=0.8, complete=False,
                       file_name='log_of_anomalies.txt'):
    """
    Receives a list of data frames, a list of anomalies
    and decides anomaly explanation based on decision tree workflow

    :param list_of_data_frames: 
    :param overall_data_frame: 
    :param categories_list: 
    :param anomaly_description: 
    :param one_day: 
    :param tolerance: 
    :return: 
    """

    print('Saving traffic report log...', end='')

    list_of_time_series = [df['views'] for df in list_of_data_frames]

    list_of_seasons = [get_one_season_average(input_series=ts, period=one_day)
                       for ts in list_of_time_series]

    overall_time_series = overall_data_frame['views']
    one_season_overall = np.array([np.median(overall_time_series[i::one_day]) for i in range(one_day)])

    traffic_change_overall = pd.Series(index=anomaly_description.index)

    for idx_anomaly in anomaly_description.index:
        overall_change = traffic_change(time_until_start=anomaly_description.loc[idx_anomaly]['largest_anomaly'],
                                                            input_series=overall_time_series,
                                                            one_season=one_season_overall)

        traffic_change_overall[idx_anomaly] = overall_change

    # Traffic change is the absolute change in number of views.
    # Percentage of change divides by the "normal" traffic at that time

    traffic_change_each_category = np.zeros(len(list_of_time_series))
    percentage_of_change_each_category = np.zeros(len(list_of_time_series))

    file = open(file_name, 'w')

    for idx_anomaly in anomaly_description.index:
        for i in range(len(list_of_time_series)):
            # width_anomaly = anomaly_description.loc[idx_anomaly]['width']
            #
            # if np.sign(anomaly_description.loc[idx_anomaly]['deviations']) > 0:
            #     point_of_interest = np.argmax(list_of_time_series[i].loc[idx_anomaly:idx_anomaly + width_anomaly])
            # else:
            #     point_of_interest = np.argmin(list_of_time_series[i].loc[idx_anomaly:idx_anomaly + width_anomaly])

            traffic_change_each_category[i] = traffic_change(time_until_start=anomaly_description.loc[idx_anomaly]['largest_anomaly'],
                                                             input_series=list_of_time_series[i],
                                                             one_season=list_of_seasons[i])
            # Calculates the percentage of change
            time_of_the_day = idx_anomaly % one_day
            # Avoids dividing by zero
            percentage_of_change_each_category[i] = traffic_change_each_category[i] / (
            1 + list_of_seasons[i][time_of_the_day])

        explanation = anomaly_decision_tree(percentage_changes_each_category=percentage_of_change_each_category,
                                            system_failure_tolerance=system_failure_tolerance)

        # Prints a log of anomalies
        time_stamp = overall_data_frame['date_time'].loc[idx_anomaly].strftime('%Y-%m-%d %H:%M:%S')

        file.write(time_stamp)
        file.write('\n')

        print_log_of_anomalies(file=file, idx_anomaly=idx_anomaly, explanation=explanation,
                               traffic_change_overall=traffic_change_overall,
                               traffic_change_each_category=traffic_change_each_category,
                               one_season_overall=one_season_overall, period=one_day,
                               list_of_categories=list_of_categories,
                               width=anomaly_description.loc[idx_anomaly]['width'], complete=complete,
                               system_failure_tolerance=system_failure_tolerance)

    print('Done!')


def print_log_of_anomalies(file, idx_anomaly, explanation, traffic_change_overall,
                           traffic_change_each_category, one_season_overall, period,
                           list_of_categories, width, system_failure_tolerance=0.8, complete=False):
    """

    :param file: 
    :param idx_anomaly: 
    :param explanation: 
    :param percentage_changes_overall: 
    :param percentage_changes_each_category: 
    :param list_of_categories: 
    :param width: 
    :param system_failure_tolerance: 
    :param complete: if true provides a complete report of anomalies, with all flag systems
    :return: 
    """

    if explanation == 'system_misbehavior':
        file.write('Possible system misbehavior.\nTraffic changed by more than %.0f%% in most sections.\n'
                   % (system_failure_tolerance * 100))
        file.write('This behaviour lasted for %d min.\n\n' % width)
    elif explanation == 'social_engagement':
        type_of_change1 = 'increased' if np.sign(traffic_change_overall[idx_anomaly]) > 0 else 'decreased'
        type_of_change2 = 'increase' if np.sign(traffic_change_overall[idx_anomaly]) > 0 else 'decrease'
        file.write('Social engagement change.\n')
        file.write('Article views %s by %.0f%% with respect to the median for this time of day.\n'
                   % (type_of_change1,
                      traffic_change_overall[idx_anomaly] / one_season_overall[idx_anomaly % period] * 100))

        cause = list_of_categories[np.argmax(traffic_change_each_category)]

        percentage_cause = np.max(traffic_change_each_category) / np.sum(traffic_change_each_category)

        file.write('Overall traffic change %d\n' % traffic_change_overall[idx_anomaly])
        file.write('Most of the change (%.0f%%) was associated with a %s in the %s section\n' % (
        percentage_cause * 100, type_of_change2, cause))
        file.write('and lasted for %d min.\n' % width)

        if complete:
            sorted_indexes = np.argsort(traffic_change_each_category)[::-1]
            for i in range(len(list_of_categories)):
                i_sorted = sorted_indexes[i]
                file.write('%s: %.0f%% change\n' % (list_of_categories[i_sorted],
                                                    100 * traffic_change_each_category[i_sorted] /
                                                    traffic_change_overall[idx_anomaly]))
                file.write('Traffic change %d\n' % traffic_change_each_category[i_sorted])

        file.write('\n')


def traffic_report(newspaper, start_date, end_date, granularity='minute', detection_granularity='day',
                   detection_period=4, explainers_of_interest=['section', 'location', 'device_type']):

    global_views = get_views_simple(newspaper=newspaper, start_date=start_date, end_date=end_date,
                                       granularity=granularity)

    global_anomalies = predict_last_period(input_data_frame=global_views, input_granularity=granularity,
                                             detection_granularity=detection_granularity, tolerance=7,
                                             detection_period=detection_period, verbose=True, clustered=True)

    # Prepares a jSON object of anomalies
    report = {}
    for [_, anomaly] in global_anomalies.iterrows():
        start = anomaly['start_date_time']
        report[start] = {}
        report[start]['end_timestamp'] = anomaly['end_date_time'].strftime(format="%Y-%m-%d %H:%M:%S")
        report[start]['largest_anomaly_in_the_period'] = anomaly['largest_anomaly']
        report[start]['duration'] = anomaly['width']
        report[start]['anomaly_type'] = ['Positive' if anomaly['deviations'] > 0 else 'Negative']
        report[start]['list_of_explainers'] = ''
        report[start]['type_of_explainers'] = ' '

    for traffic_source in explainers_of_interest:
        print('\rAnalysing %s' % traffic_source)
        traffic_source_views = get_views_column(newspaper=newspaper, start_date=start_date, end_date=end_date,
                                                traffic_source=traffic_source,granularity=granularity)

        categorised_traffic_source = split_by_category(input_data_frame=traffic_source_views,
                                                       column_name='traffic_source')

        categorised_anomalies = get_categorised_anomalies(categorised_time_series=categorised_traffic_source,
                                                          input_granularity=granularity,
                                                          detection_granularity=detection_granularity,
                                                          detection_period=detection_period)

        explainers = list_of_explainers(global_anomalies=global_anomalies, categorised_anomalies=categorised_anomalies)

        for [_, anomaly] in explainers.iterrows():
            # If not most of the traffic sources are detected as anomalous, flag this as possible explainer
            if len(anomaly['explainer'])/len(categorised_traffic_source) < 0.7:
                start = anomaly['start_date_time']
                explainer = ', '.join(anomaly['explainer'])
                report[start]['list_of_explainers'] = report[start]['list_of_explainers'] + explainer + ','
                report[start]['type_of_explainers'] = report[start]['type_of_explainers'] + ',' + traffic_source


    return json.dumps(report)