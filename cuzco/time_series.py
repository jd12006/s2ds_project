import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta

sns.set()
sns.set_context("paper", rc={"lines.linewidth": 0.5})


def time_series_decomposition(input_series, method='Median Removal', window=100, periodicity=1440, plot=False):
    """
       Splits a time series in three components.
    
       The moving-average window and periodicity are adjusted for a minute-basis detection. 
    
       :param input_series: pandas time series based on page views per day, hour or minute.
       :param method: a string specifying the decomposition method. 'Median removal' or 'Additive'
       :param window: an integer representing the window of the Moving Average, in the units of the input_series granularity.
       :param periodicity: length of the period to evaluate the seasonal component, in the units of the input_series granularity.
       :param plot: whether to plot the time series decomposition.
    
       :return [trend, seasonal, residual] list with three pandas series.
    
       """

    if method == 'Additive':
        # Estimates trend using Moving Average filter and model
        # X = T+S+R, where T is the trend component, S is the seasonality and R is the residual series.

        trend_component = input_series.rolling(window=window).median()[window - 1:]

        detrended_series = input_series[window - 1:] - trend_component

        # Calculates the seasonal component first as an np_array
        one_season = np.array([np.median(detrended_series[i::periodicity]) for i in range(periodicity)])
        number_of_periods = int(len(detrended_series) / periodicity)
        remainder = len(detrended_series) % periodicity

        season_component = np.tile(one_season, number_of_periods)

        # Makes sure that the season component has the correct size
        if remainder:
            season_component = np.append(season_component, season_component[:remainder])

        # Converts season component to pd.series

        season_component = pd.Series(data=season_component, index=trend_component.index)

        residual_component = input_series[window - 1:] - trend_component - season_component

        if plot:
            x = np.arange(len(input_series))
            plt.subplot(4, 1, 1);
            plt.title('Series')
            plt.plot(x, input_series)
            plt.subplot(4, 1, 2);
            plt.title('Trend Component')
            plt.plot(x[window - 1:], trend_component);
            plt.subplot(4, 1, 3);
            plt.title('Seasonal Component')
            plt.plot(x[window - 1:], season_component);
            plt.subplot(4, 1, 4);
            plt.title('Residual Component')
            plt.plot(x[window - 1:], residual_component);
            plt.tight_layout()

    elif method == 'Median Removal':

        trend_component = input_series.median() * np.ones(len(input_series))

        detrended_series = input_series - trend_component

        # Calculates the seasonal component first as an np_array
        one_season = np.array([np.median(detrended_series[i::periodicity]) for i in range(periodicity)])
        number_of_periods = int(len(detrended_series) / periodicity)
        remainder = len(detrended_series) % periodicity

        season_component = np.tile(one_season, number_of_periods)

        # Makes sure that the season component has the correct size
        if remainder:
            season_component = np.append(season_component, season_component[:remainder])

        # Converts season component to pd.series

        season_component = pd.Series(data=season_component, index=input_series.index)

        residual_component = input_series - trend_component - season_component

        if plot:
            x = np.arange(len(input_series))
            plt.subplot(4, 1, 1);
            plt.title('Series')
            plt.plot(x, input_series)
            plt.subplot(4, 1, 2);
            plt.title('Trend Component')
            plt.plot(x, trend_component);
            plt.subplot(4, 1, 3);
            plt.title('Seasonal Component')
            plt.plot(x, season_component);
            plt.subplot(4, 1, 4);
            plt.title('Residual Component')
            plt.plot(x, residual_component);
            plt.tight_layout()

    return [trend_component, season_component, residual_component]


def global_anomaly_detection(input_data_frame, threshold_type='Median', tolerance='Auto',
                             method='Median Removal', window=None, periodicity=None, verbose=False):
    """
    Main module: detects anomalies.
    
    TODO: implement threshold type standard deviation from average.
    
    :param input_data_frame: pandas dataframe of page views by day, hour or minute.
    :param threshold_type: type of threshold. 'std' or 'median'.
    :param tolerance: amount of deviation from the median tolerated before a point is considered anomalous; default = "Auto" for auto-tuning; a lower tolerance will flag more anomalies.
    :param method: a string specifying the decomposition method. Only 'Naive' allowed up to now.
    :param window: an integer representing the window of the Moving Average, in the units of the input_data_frame granularity.
    :param periodicity: length of the period to evaluate the seasonal component, in the units of the input_data_frame granularity.
    
    :return: pandas dataframe with the time and deviation of anomalies.
    
    """

    input_series = input_data_frame['views']

    residual_series = time_series_decomposition(input_series=input_series,
                                                method=method, window=window, periodicity=periodicity)[2]
    if threshold_type == 'Median':
        
        # Grid search: Still not a good approach to tune tolerance
        if tolerance == 'Auto':
            if verbose:
                print('Auto tolerance tuning on...')
            tolerances = np.arange(4, 13, 0.1)

            best_tolerance = 13
            previously_detected_anomalies = len(residual_series)

            for i in range(len(tolerances)):

                anomalous_points = find_mad_deviation(input_series=residual_series, tolerance=tolerances[i])
                number_of_detected_anomalies = len(anomalous_points)

                if abs(previously_detected_anomalies-number_of_detected_anomalies) <= 1:
                    best_tolerance = tolerances[i]
                    break
                else:
                    previously_detected_anomalies = number_of_detected_anomalies

            if verbose:
                print('Optimal tolerance: %.1f' % best_tolerance)

        else:
            if verbose:
                print('Using tolerance %.2f' % tolerance)
            anomalous_points = find_mad_deviation(input_series=residual_series, tolerance=tolerance)

    indexes = anomalous_points.index

    anomaly_data_frame = input_data_frame.loc[indexes]
    anomaly_data_frame['deviations'] = anomalous_points

    return anomaly_data_frame


def find_mad_deviation(input_series, tolerance=7):
    """
    Find the points of a time series that deviates from baseline MAD.
     
    :param input_series: pandas time series, based on page views per day, hour or minute.
    :param tolerance: amount of deviation from the median tolerated before a point is considered anomalous; a lower tolerance will flag more anomalies.
    
    :return: time series with the points which deviated from normal and corresponding deviations.

    """

    median = np.median(input_series)
    mad = np.median(np.abs(input_series - median))

    # Calculate the anomalous points according to the tolerance rule

    deviations = input_series-median

    anomalous_points = deviations[np.abs(deviations) > tolerance * mad]

    return anomalous_points


def cluster_anomalies(anomaly_data_frame, step=2):
    """
    Clusters consecutive anomalies that occured in the same (positive or negative) direction.
    Outputs a pandas data frame consisting of:
        * anomaly with largest deviation in the period
        * start time
        * end time
    
    :param anomaly_data_frame: pandas dataframe of anomalies with the time, views, deviation and width (if clustered=True) of each anomaly.
    :param step: a step that defines "consecutive" points; default=2.
    
    :return pandas dataframe, 2 columns with clustered anomalies and width. 
    
    """

    time_stamps = anomaly_data_frame.index
    directions = np.sign(anomaly_data_frame['deviations']).get_values()

    cluster_anomalies_indices_list = []
    list_of_widths = []
    list_of_worse_anomalies = []
    list_of_views_of_worse_anomalies = []
    list_of_end_anomalies = []

    if len(time_stamps) > 0:
        index_of_start_of_anomaly = time_stamps[0]
        views_of_worse_anomaly = anomaly_data_frame.loc[index_of_start_of_anomaly]['views']
        index_of_worse_anomaly = time_stamps[0]
        deviation_of_worse_anomaly = anomaly_data_frame.loc[index_of_worse_anomaly]['deviations']

        cluster_anomalies_indices_list = [index_of_start_of_anomaly]

        current_width = 1
        # Index variable to check consecutive directions
        index = 1
        previous_index = index_of_start_of_anomaly

        for stamp in time_stamps[1:]:
            sign_of_anomaly = directions[index]
            if ((stamp - previous_index) > step) or (directions[index] != directions[index-1]):
                cluster_anomalies_indices_list += [stamp]
                list_of_widths += [current_width]
                list_of_views_of_worse_anomalies += [views_of_worse_anomaly]
                list_of_worse_anomalies += [index_of_worse_anomaly]

                list_of_end_anomalies += [previous_index]

                current_width = 1
                views_of_worse_anomaly = anomaly_data_frame.loc[stamp]['views']
                deviation_of_worse_anomaly = anomaly_data_frame.loc[stamp]['deviations']

            else:
                current_width += 1
                if sign_of_anomaly > 0:
                    if anomaly_data_frame.loc[stamp]['deviations'] > deviation_of_worse_anomaly:
                        index_of_worse_anomaly = stamp
                        views_of_worse_anomaly = anomaly_data_frame.loc[stamp]['views']
                        deviation_of_worse_anomaly = anomaly_data_frame.loc[stamp]['deviations']
                else:
                    if anomaly_data_frame.loc[stamp]['deviations'] < deviation_of_worse_anomaly:
                        index_of_worse_anomaly = stamp
                        views_of_worse_anomaly = anomaly_data_frame.loc[stamp]['views']
                        deviation_of_worse_anomaly = anomaly_data_frame.loc[stamp]['deviations']

            previous_index = stamp
            index += 1

        # Updates the remaning element of the loop
        list_of_widths += [current_width]
        list_of_worse_anomalies += [index_of_worse_anomaly]
        list_of_views_of_worse_anomalies += [views_of_worse_anomaly]

    anomaly_clustered_data_frame = anomaly_data_frame.loc[cluster_anomalies_indices_list]
    anomaly_clustered_data_frame = anomaly_clustered_data_frame.rename(columns={'date_time':'start_date_time'})

    anomaly_clustered_data_frame.loc[:, 'width'] = np.array(list_of_widths)
    anomaly_clustered_data_frame.loc[:, 'views'] = np.array(list_of_views_of_worse_anomalies)
    anomaly_clustered_data_frame.loc[:, 'largest_anomaly'] = np.array(list_of_worse_anomalies)

    # Drop the anomalies which are artificially introduced by the fill_time_series_with_zeros modules
    anomaly_clustered_data_frame = anomaly_clustered_data_frame[anomaly_clustered_data_frame['date'] != 0]

    if len(anomaly_clustered_data_frame)>0:
        anomaly_clustered_data_frame = insert_date_time(anomaly_clustered_data_frame)
        anomaly_clustered_data_frame = insert_end_date_time(anomaly_clustered_data_frame)

    return anomaly_clustered_data_frame


def predict_last_period(input_data_frame, input_granularity='minute', detection_granularity='day',
                        detection_period=5, window=100, periodicity=1440,
                        tolerance=7, clustered=True, verbose=False):
    """
    Receives an input data frame and outputs anomalous data points detected within the last period_window.

    :param input_data_frame: pandas dataframe of page views by day, hour or minute.
    :param input_granularity: granularity of input_data_frame (day, hour or minute).
    
    :param detection_granularity: whether detection_period units are days, hours or minutes.
    :param detection_period: retroactive period over which to detect anomalies.
    
    :param window: an integer representing the window of the Moving Average, in the units of the input_granularity (defaults are 1 day, 2 hours or 100 minutes for input_granularity = day, hour or minute).
    :param periodicity: length of the period to evaluate the seasonal component, in the units of the input_granularity.
    
    :param tolerance: amount of deviation from the median tolerated before a point is considered anomalous; default = "Auto" for auto-tuning; a lower tolerance will flag more anomalies.
    :param clustered: whether to cluster consecutive anomalies together.
    :param verbose: whether to display the anomalies on the screen.
    
    :return: pandas dataframe of anomalies with the time, views, deviation and width (if clustered=True) of each anomaly.
    
    """
    
    if verbose:
        start_time = time.time()

        
    ### Flexibility to accept parameters given in different granularities:
    ### converts all parameter units to granularity of input data frame.
    
    if input_granularity == "day":
        if window is None:
            window = 1
        if periodicity is None:
            periodicity = 1
        if detection_granularity == "hour" or detection_granularity == "minute":
            print("Error: detection_granularity is smaller than input_granularity")
            return
            
    elif input_granularity == "hour":
        if window is None:
            window = 2
        if periodicity is None:
            periodicity = 24
        if detection_granularity == "day":
            detection_period = detection_period*24
        elif detection_granularity == "minute":
            print("Error: detection_granularity is smaller than input_granularity") 
            return
            
    elif input_granularity == "minute":
        if window is None:
            window = 200
        if periodicity is None:
            periodicity = 1440
        if detection_granularity == "day":
            detection_period = detection_period*1440
        elif detection_granularity == "hour":
            detection_period = detection_period*60

    ### Anomaly detection
    anomaly_data_frame = global_anomaly_detection(input_data_frame = input_data_frame,
                                                   tolerance = tolerance,
                                                   window = window,
                                                   periodicity = periodicity,
                                                   verbose = verbose)
    
    final_period = input_data_frame.index[-1]-detection_period
    
    indexes_for_detection = (anomaly_data_frame.index > final_period)
    
    anomalies_in_the_desired_period = anomaly_data_frame[indexes_for_detection]
    
    if verbose:
        print('**************\nSummary Report \n**************')
        percentage_of_anomalies = float(len(anomalies_in_the_desired_period)/detection_period)
        print('Percentage of anomalies in the period: %.3f%%' % (100*percentage_of_anomalies))
        end_time = time.time()
        print('Time for detection: %.3f seconds' % (end_time - start_time))
    
    if clustered:
        cluster = cluster_anomalies(anomalies_in_the_desired_period)
    
        return cluster
    
    else:
        return anomalies_in_the_desired_period


def plot_detected_anomalies(input_data_frame, anomaly_data_frame, 
                            input_granularity, detection_granularity,
                            detection_period, whole_period_days=14, show_anomalies = True, save=False,
                            plot_range='All', ticks='All', labels='All', center_clusters=False):
    """ 
    Plots the anomalies found standard period set for a minutely granularity and two-week analysis.
    For now, it only works for the minuetly analysis. 
    
    :param input_data_frame: pandas dataframe of page views by day, hour or minute.
    :param anomaly_data_frame: pandas dataframe of anomalies with time, views, deviation and width (if clustered) - at same granularity as input_data_frame.
    
    :param input_granularity: whether input_data_frame is in days, hours or minutes.
    :param detection_granularity: whether detection_period units are days, hours or minutes.
    
    :param detection_period: retroactive period over which to detect (and highlight) anomalies.    
    :param whole_period_days: number of days in the input_data_frame (default=14).
    
    :param save: whether to save the plot in the /validation/ folder.
    
    """
    
    ### Flexibility to work with different granularities
    continue_procedure = True
    one_day = 1
    
    if input_granularity == "day":
        if detection_granularity == "hour" or detection_granularity == "minute":
            print("Error: detection_granularity is smaller than input_granularity")   
            continue_procedure = False
            
    elif input_granularity == "hour":
        one_day = 24
        if detection_granularity == "day":
            detection_period = detection_period*24
        elif detection_granularity == "minute":
            print("Error: detection_granularity is smaller than input_granularity") 
            continue_procedure = False
            
    elif input_granularity == "minute":
        one_day = 1440
        if detection_granularity == "day":
            detection_period = detection_period*1440
        elif detection_granularity == "hour":
            detection_period = detection_period*60
            
    whole_period_days = whole_period_days*one_day


    ### Draw the plot       
    if continue_procedure == False:
        return
    else:
        anomalies = anomaly_data_frame.index
    
        # Past_series plot
        input_series = input_data_frame['views']
        x = input_data_frame.index
    
        past_series = input_data_frame[:(whole_period_days - detection_period)]['views'].get_values()
        plt.plot(x[:whole_period_days - detection_period], past_series, alpha=0.3);
    
        # Current day plot
        final_series = input_data_frame[(whole_period_days - detection_period):whole_period_days]['views'].get_values()
        plt.plot(x[whole_period_days - detection_period:whole_period_days], final_series, alpha=1, color='b');

        ### Anomalies
        if center_clusters:
            anomalies = [idx + int(anomaly_data_frame.loc[idx]['width']/2) for idx in anomalies]
        if show_anomalies:
            plt.plot(anomalies, input_series[anomalies], 'ro', alpha=0.8)
    
        # Ticks and labels
        xticks_labels = input_data_frame[:whole_period_days + detection_period]['date'][::one_day].apply(lambda str: str[-5:]);
        xticks = x[range(whole_period_days)[::one_day]]
    
        plt.ylim([0-0.01*np.max(input_series), 1.4*np.max(input_series)])
        plt.ylabel('Number of Views');
    
        if plot_range != 'All':
            plt.xlim([plot_range[0], plot_range[1]])
        else:
            plt.xlim([x[0], x[-1]])
    
        if ticks != 'All':
            plt.xticks(ticks, labels, rotation=60)
        else:
            xticks_labels = input_data_frame[:whole_period_days + detection_period]['date'][::one_day].apply(
                lambda str: str[-5:]);
            xticks = x[range(whole_period_days)[::one_day]]
            plt.xticks(xticks, xticks_labels, rotation=60);
    
        if save:
            plt.savefig('data/'+save, dpi=1000)


#######################################################################
# Convenience functions to add timestamp columns to pandas dataframes #
#######################################################################

def extract_date_time(d0):
    """ 
    Concatenates date, hour and minute into a timestamp.
    d0 = dataframe of anomalies containing columns for date, hour, minute and width.
    """
    return datetime.strptime(d0['date']+' '+str(d0['hour'])+' '+str(d0['minute']),'%Y-%m-%d %H %M')


def insert_date_time(input_data_frame):
    """
    Called in predict_last_period().
    Inserts a column with date times (notice no t0 parameter in this one).
    Assumes that time is in minutes.
    
    :param input_data_frame: pandas dataframe with day, hour and minute columns.
    :return: input_data_frame with new column containing datetime rounded to minutes.  
    """
    input_data_frame['date_time'] = ''
    input_data_frame['date_time'] = input_data_frame[['date', 'hour', 'minute']].apply(extract_date_time, axis=1) 
    
    return input_data_frame


def extract_end_date_time(d0):
    """ 
    Concatenates date, hour and minute into a timestamp and adds the width (assumed to be in minutes).
    d0 = dataframe of anomalies containing columns for date, hour, minute and width.
    """
    return datetime.strptime(d0['date']+' '+str(d0['hour'])+' '+str(d0['minute']),'%Y-%m-%d %H %M') + timedelta(minutes = d0['width'])


def insert_end_date_time(input_data_frame):
    """
    Inserts columns with date times. 
    Assumes that time is in minutes.
    
    :param input_data_frame: pandas dataframe with day, hour and minute columns.
    :return: input_data_frame with new column containing datetime rounded to minutes.
    """
    input_data_frame['end_date_time'] = ''
    input_data_frame['end_date_time'] = input_data_frame[['date', 'hour', 'minute', 'width']].apply(extract_end_date_time, axis=1) 
    
    return input_data_frame
