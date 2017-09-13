
# coding: utf-8

""" Modules to get page view data from BigQuery """

import pandas as pd
import numpy as np
import os
from cuzco import time_series
from datetime import datetime, timedelta


def get_bytes():
    """
    Calculates total size of dataset in gigabytes.
    """

    project_id = "newsuk-datatech-s2ds-2017-2"
            
    query = """
            SELECT
              SUM(size_bytes)/pow(10,9) AS gigabytes
            FROM
              [newsuk-datatech-s2ds-2017-2:project_data.__TABLES__]
            """
        
    return pd.io.gbq.read_gbq(query = query, project_id = project_id)


# Get table with time frame and views only - no extra columns are created.

def get_views_simple(newspaper, start_date, end_date, granularity="day"):
    """
    Returns total page views per day for a given newspaper within a given date range.
    Saves a CSV file.
    
    Input parameters:
    newspaper = "sun" or "times"
    start_date, end_date = date range to include records from (strings formatted as 'YYYY-MM-DD')
    granularity = "day", "hour" or "minute" (default is day)
    """
    
    project_id = "newsuk-datatech-s2ds-2017-2"
    
    if granularity == "day":
        time_frame1 = "DATE(activity_date) AS date"
        time_frame2 = "date"
        sort_fields = ['date']
    elif granularity == "hour":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour"
        time_frame2 = "date, hour"
        sort_fields = ['date','hour']
    elif granularity == "minute":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour, MINUTE(activity_date_time) AS minute"
        time_frame2 = "date, hour, minute"
        sort_fields = ['date','hour','minute']
            
    query = """
            SELECT
              {},
              SUM(page_views) AS views
            FROM
              TABLE_DATE_RANGE([project_data.{}_data_],
              TIMESTAMP('{}'),
              TIMESTAMP('{}'))
            WHERE
              content_id IS NOT NULL
            GROUP BY
              {}
            ORDER BY
              {}
            """
    query = query.format(time_frame1, newspaper, start_date, end_date, time_frame2, time_frame2)
    
    filename = 'data/views_{}_by_{}_{}_{}.csv'.format(newspaper, granularity, start_date, end_date)
    print("Creating...", filename)
    
    if os.path.exists(filename):
        print("File already exists!")
        views_df = pd.read_csv(filename, index_col = 0)
        return views_df
    else:
        views_df = pd.io.gbq.read_gbq(query = query, project_id = project_id)       
        views_df = views_df.sort_values(by = sort_fields)
        views_df = views_df.reset_index(drop = True)
        
        if granularity=="minute":
            views_df = time_series.insert_date_time(views_df)
        
        views_df.to_csv(filename)
        
        return views_df


def get_views_simple_weeks_before(newspaper, weeks_before, end_date, granularity="day"):
    """
    Alternative version of get_views_simple.
    Returns total page views per day for dates between a given end_date and a
    given number of weeks before that date.
    Saves a CSV file.
    
    Input parameters:
    newspaper = "sun" or "times"
    weeks_before = integer, number of weeks of data to get before the end date
    end_date = last date in range to include records from (string formatted as 'YYYY-MM-DD')
    granularity = "day", "hour" or "minute" (default is day)
    """
    
    project_id = "newsuk-datatech-s2ds-2017-2"
    
    if granularity == "day":
        time_frame1 = "DATE(activity_date) AS date"
        time_frame2 = "date"
        sort_fields = ['date']
    elif granularity == "hour":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour"
        time_frame2 = "date, hour"
        sort_fields = ['date','hour']
    elif granularity == "minute":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour, MINUTE(activity_date_time) AS minute"
        time_frame2 = "date, hour, minute"
        sort_fields = ['date','hour','minute']
            
    query = """
            SELECT
              {},
              SUM(page_views) AS views
            FROM
              TABLE_DATE_RANGE([project_data.{}_data_],
              DATE_ADD(TIMESTAMP('{}'),(1-({}*7)),'DAY'),
              DATE_ADD(TIMESTAMP('{}'),0,'DAY'))
            WHERE
              content_id IS NOT NULL
            GROUP BY
              {}
            ORDER BY
              {}
            """
    query = query.format(time_frame1, newspaper, end_date, weeks_before, end_date, time_frame2, time_frame2)

    filename = 'data/views_{}_by_{}_{}_-{}weeks.csv'.format(newspaper, granularity, end_date, weeks_before)
    print("Creating...", filename)
    
    if os.path.exists(filename):
        print("File already exists!")
        views_df = pd.read_csv(filename, index_col = 0)
        return views_df
    else:
        views_df = pd.io.gbq.read_gbq(query = query, project_id = project_id)       
        views_df = views_df.sort_values(by=sort_fields)
        views_df = views_df.reset_index(drop=True)
        
        if granularity=="minute":
            views_df = time_series.insert_date_time(views_df)

        views_df.to_csv(filename)
        
        return views_df


# Get table with time frame and views by location

def get_views_location(newspaper, start_date, end_date, granularity="day"):
    """
    Returns total page views per day for a given newspaper within a given date range, 
    either in Great Britain (GBR) or the Rest of the World (ROW).
    Saves a CSV file.
    
    Input parameters:
    newspaper = "sun" or "times"
    location = "GBR" or "ROW"
    start_date, end_date = date range to include records from (strings formatted as 'YYYY-MM-DD')
    granularity = "day", "hour" or "minute" (default is day)
    """
    
    project_id = "newsuk-datatech-s2ds-2017-2"
    
    if granularity == "day":
        time_frame1 = "DATE(activity_date) AS date"
        time_frame2 = "date"
        sort_fields = ['date']
    elif granularity == "hour":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour"
        time_frame2 = "date, hour"
        sort_fields = ['date','hour']
    elif granularity == "minute":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour, MINUTE(activity_date_time) AS minute"
        time_frame2 = "date, hour, minute"
        sort_fields = ['date','hour','minute']
            
    query = """
            SELECT
              {},
              IF(country == "GBR", "GBR", "ROW") AS location,
              SUM(page_views) AS views
            FROM
              TABLE_DATE_RANGE([project_data.{}_data_],
              TIMESTAMP('{}'),
              TIMESTAMP('{}'))
            WHERE
              content_id IS NOT NULL
            GROUP BY
              {}, location
            ORDER BY
              {}
            """
    query = query.format(time_frame1, newspaper, start_date, end_date, time_frame2, time_frame2)

    filename = 'data/views_{}_by_{}_location_{}_{}.csv'.format(newspaper, granularity, start_date, end_date)
    print("Creating...", filename)
    
    if os.path.exists(filename):
        print("File already exists!")
        views_df = pd.read_csv(filename, index_col = 0)
        return views_df
    else:
        views_df = pd.io.gbq.read_gbq(query = query, project_id = project_id)       
        views_df = views_df.sort_values(by=sort_fields)
        views_df = views_df.reset_index(drop=True)
        views_df = views_df.rename(index=str, columns={"location": "traffic_source"})
        
        if granularity=="minute":
            views_df = time_series.insert_date_time(views_df)

        views_df.to_csv(filename)
        
        return views_df


def get_views_section(newspaper, start_date, end_date, granularity="day", clean=True):
    """
    Returns total page views per day for a given newspaper within a given date range, 
    subsetted into article sections, e.g. sport, news, travel. 
    Saves a CSV file.
    
    Input parameters:
    newspaper = "sun" or "times" 
    start_date, end_date = date range to include records from (strings formatted as 'YYYY-MM-DD')
    granularity = "day", "hour" or "minute" (default is day)
    clean = True or False - whether to clean the sections to return only the main ones
    """
    
    project_id = "newsuk-datatech-s2ds-2017-2"
    
    if granularity == "day":
        time_frame1 = "DATE(activity_date) AS date"
        time_frame2 = "date"
        sort_fields = ['date']
    elif granularity == "hour":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour"
        time_frame2 = "date, hour"
        sort_fields = ['date','hour']
    elif granularity == "minute":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour, MINUTE(activity_date_time) AS minute"
        time_frame2 = "date, hour, minute"
        sort_fields = ['date','hour','minute']
    
    if clean:
        clean_label = "clean"
    else:
        clean_label = "notclean"
            
    query = """
            SELECT
              {},
              IF(section == "page3", "page 3", 
                  IF(section LIKE("%tv%"), "tv & showbiz", 
                      IF(section LIKE("%showbiz%"), "tv & showbiz", section))) AS section,
              SUM(page_views) AS views
            FROM
              TABLE_DATE_RANGE([project_data.{}_data_],
              TIMESTAMP('{}'),
              TIMESTAMP('{}'))
            WHERE
              content_id IS NOT NULL
            GROUP BY
              {}, section
            ORDER BY
              {}, section
            """
    query = query.format(time_frame1, newspaper, start_date, end_date, time_frame2, time_frame2)
        
    filename = 'data/views_{}_by_{}_section_{}_{}_{}.csv'.format(newspaper, granularity, clean_label, start_date, end_date)
    print("Creating...", filename)
    
    if os.path.exists(filename):
        print("File already exists!")
        views_df = pd.read_csv(filename, index_col = 0)
        return views_df
    else:
        views_df = pd.io.gbq.read_gbq(query = query, project_id = project_id)        
        views_df = views_df.sort_values(by=sort_fields)
        views_df = views_df.reset_index(drop=True)
        
        if granularity=="minute":
            views_df = time_series.insert_date_time(views_df)
        
        if clean:
            print('Cleaning sections...')
            views_df = clean_sections(views_df, newspaper = newspaper, granularity = granularity)
            print("...Sections were cleaned!")
        else:
            print("You can reduce the number of sections using clean_sections")
        
        views_df.to_csv(filename)
        
        return views_df

# Get views grouped by a specific column

def get_views_column(newspaper, start_date, end_date, traffic_source='product', granularity="day", clean=None):
    """
    Returns total page views per day for a given newspaper within a given date range, 
    either in Great Britain (GBR) or the Rest of the World (ROW).
    Saves a CSV file.

    Input parameters:
    newspaper = "sun" or "times"
    traffic_source = desired column of the newspaper
    start_date, end_date = date range to include records from (strings formatted as 'YYYY-MM-DD')
    granularity = "day", "hour" or "minute" (default is day)
    clean = True/False (default=None) - only used if the traffic_source is 'section'.
    """
    
    cleanlab = ""
    
    if traffic_source == "location":
        views_df = get_views_location(newspaper, start_date, end_date, granularity)
        views_df = views_df.rename(columns={'location' : 'traffic_source'})
        return views_df
    
    elif traffic_source == "section":
        if clean == False:
            cleanlab  = "_notclean"
        else:
            clean = True
            cleanlab = "_clean"

        views_df = get_views_section(newspaper, start_date, end_date, granularity, clean=clean)
        views_df = views_df.rename(columns={'section': 'traffic_source'})
        return views_df
    
    else:
        project_id = "newsuk-datatech-s2ds-2017-2"
    
        if granularity == "day":
            time_frame1 = "DATE(activity_date) AS date"
            time_frame2 = "date"
            sort_fields = ['date']
        elif granularity == "hour":
            time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour"
            time_frame2 = "date, hour"
            sort_fields = ['date','hour']
        elif granularity == "minute":
            time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour, MINUTE(activity_date_time) AS minute"
            time_frame2 = "date, hour, minute"
            sort_fields = ['date','hour','minute']
    
        query = """
                SELECT
                  {},
                  {} AS traffic_source,
                  SUM(page_views) AS views
                FROM
                  TABLE_DATE_RANGE([project_data.{}_data_],
                  TIMESTAMP('{}'),
                  TIMESTAMP('{}'))
                WHERE
                  content_id IS NOT NULL
                GROUP BY
                  {}, traffic_source
                ORDER BY
                  {}
                """
        query = query.format(time_frame1, traffic_source, newspaper, start_date, end_date, time_frame2, time_frame2)
    
        filename = 'data/views_{}_by_{}_{}{}_{}_{}.csv'.format(newspaper, granularity, traffic_source, cleanlab, start_date, end_date)
        print("Creating...", filename)
        
        if os.path.exists(filename):
            print("File already exists!")
            views_df = pd.read_csv(filename, index_col = 0)
            return views_df
        else:
            views_df = pd.io.gbq.read_gbq(query = query, project_id = project_id)       
            views_df = views_df.sort_values(by=sort_fields)
            views_df = views_df.reset_index(drop=True)

            if granularity=="minute":
                views_df = time_series.insert_date_time(views_df)
                
            views_df.to_csv(filename)
            
            return views_df


# Get views with extra columns to help explain unusual behaviour in page view counts (but not used in cuzco package...)

def get_views_extra(newspaper, start_date, end_date, granularity="day"):
    """
    Returns total page views per day for a given newspaper within a given date range.
    Plus all the other columns we should need for the MVP. 
    Saves a CSV file.
    
    Included day of week (1=Mon, 7=Sun).
    Location: GBR (Great Britain) or ROW (Rest of World).
    Used if statements for some data cleaning.
    
    Input parameters:
    newspaper = "sun" or "times"
    start_date, end_date = date range to include records from (strings formatted as 'YYYY-MM-DD')
    granularity = "day", "hour" or "minute" (default is day)
    """
    
    project_id = "newsuk-datatech-s2ds-2017-2"
    
    if granularity == "day":
        time_frame1 = "DATE(activity_date) AS date"
        time_frame2 = "date"
        sort_fields = ['date']
    elif granularity == "hour":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour"
        time_frame2 = "date, hour"
        sort_fields = ['date','hour']
    elif granularity == "minute":
        time_frame1 = "DATE(activity_date) AS date, HOUR(activity_date_time) AS hour, MINUTE(activity_date_time) AS minute"
        time_frame2 = "date, hour, minute"
        sort_fields = ['date','hour','minute']
            
    query = """
            SELECT
              {},
              DAYOFWEEK(activity_date_time) AS day_of_week,
              IF(country == "GBR", "GBR", "ROW") AS location,
              IF(device_type == "UNKNOWN", "Browser", device_type) AS device,
              referrer_type AS referrer,
              product,
              IF(section == "page3", "page 3", 
                  IF(section LIKE("%tv%"), "tv & showbiz", 
                      IF(section LIKE("%showbiz%"), "tv & showbiz", section))) AS section,
              SUM(page_views) AS total_views
            FROM
              TABLE_DATE_RANGE([project_data.{}_data_],
              TIMESTAMP('{}'),
              TIMESTAMP('{}'))
            WHERE
              content_id IS NOT NULL
            GROUP BY
             {}, day_of_week, location, device, referrer, product, section
            ORDER BY
             {}
            """
    
    query = query.format(time_frame1, newspaper, start_date, end_date, time_frame2, time_frame2)
        
    filename = 'data/views_{}_by_{}_extra_columns_{}_{}.csv'.format(newspaper, granularity, start_date, end_date)
    print("Creating...", filename)
    
    if os.path.exists(filename):
        print("File already exists!")
        views_df = pd.read_csv(filename, index_col = 0)
        return views_df
    else:
        views_df = pd.io.gbq.read_gbq(query = query, project_id = project_id)       
        views_df = views_df.sort_values(by=sort_fields)
        views_df = views_df.reset_index(drop=True)

        if granularity=="minute":
            views_df = time_series.insert_date_time(views_df)

        views_df.to_csv(filename)

        return views_df


def clean_sections(dataframe, newspaper, granularity="minute"): 
    """
    Condenses article sections into key sections only and labels the remaining 
    ones as 'other'. 
    
    Input parameters: 
    dataframe = pandas dataframe with a column called section.
    newspaper = "sun" or "times". 
    granularity = day, hour or minute (default minute).
    """
    sun_sections = ['news', 'living', 'money', 'motors', 'sport', 'tv & showbiz', 'tech', 'travel', 'all']
    times_sections = ['news', 'law', 'money', 'sport', 'the game', 'world', 'business', 'register',
                      'comment', 'puzzles', 'scotland', 'ireland', 'news review', 'saturday review',
                      'times2', 'the times magazine', 'the sunday times magazine', 'weekend', 'culture',  
                      'home', 'travel', 'style', 'the dish', 'bricks & mortar']  ### Excluded 'rich list' and 'irish rich list 2017'
    
    if newspaper == "sun":
        section_list = sun_sections
    elif newspaper == "times":
        section_list = times_sections

    dataframe_clean = dataframe.copy()
    to_replace = {}

    # A hash to map rename the section in time O(n)
    for section in dataframe_clean['section'].unique():
        if section in section_list:
            to_replace[section] = section
        else:
            to_replace[section] = 'other'

    dataframe_clean['section'] = dataframe_clean['section'].map(to_replace)

    # Old code
    # dataframe_clean = dataframe.copy()
    #
    # to_replace = [section for section in dataframe_clean['section'] if section not in section_list]
    # value = 'other'
    # dataframe_clean['section'].replace(to_replace, value, inplace=True)
    
    # aggregate rows to ensure there is only one row per section per day, hour or minute:
    if granularity == "day":
        dataframe_clean = dataframe_clean.groupby(['date', 'section']).aggregate(sum)
    elif granularity == "hour":
        dataframe_clean = dataframe_clean.groupby(['date', 'hour', 'section']).aggregate(sum)
    elif granularity == "minute":
        dataframe_clean = dataframe_clean.groupby(['date', 'hour', 'minute', 'section']).aggregate(sum)
    
    dataframe_clean = dataframe_clean.reset_index()
    
    return dataframe_clean


def mark_holidays(dataframe):
    """    
    Marks dates as True if they are bank holidays in UK and False if not, in a 
    new column called 'holiday'.
    
    Could extend this to consider holidays in whichever country the view was made.
    
    Requires the holidays module.
    
    Input parameters:
    dataframe = a pandas dataframe containing a date column
    """
    import holidays
    uk_holidays = holidays.UK()  
    column = []
    for date in dataframe['date']:
        if date in uk_holidays:
            x = "True"
        else:
            x = "False"
        column.append(x)
    dataframe['holiday'] = column
