# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:33:36 2017

### MODULE TO GET ARTICLES VIEWED DURING AN ANOMALY
"""

from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import numpy as np

#Get articles by section 

def get_articles_contents_by_section(newspaper, section, start, end, percentage = 0.9):
    """
    Returns the ID, headline and content of all articles viewed during an anomaly.
    Creates a directory article_contents_section_start_end/ 
    and save each file in a txt.
    
    Input parameters:
    newspaper = "sun" or "times"
    section = the section to return articles from (? what about the ALL section...)
    start, end = start and end datetime of anomaly (strings formatted as 'YYYY-MM-DD HH:MM:SS')
    percentage = only gets the articles that represent more than 90% of the views.
    """
    
    project_id = "newsuk-datatech-s2ds-2017-2"

    # Start calculates the date.
    start_date = start.split()[0]
    start_label = start.replace(":", "-")
    start_label = start_label.replace(" ", "_")
    
    end_date = end.split()[0] 
    end_label = end.replace(":", "-")
    end_label = end_label.replace(" ", "_")
    
    query = """
    SELECT content_id, headline, content
    FROM (
    SELECT content_id, percentage, SUM(percentage)
        OVER 
        (ORDER BY percentage DESC) as running,
        headline, content, views
    FROM (
    SELECT
      click_data.content_id AS content_id,
      MAX(click_data.views) AS views,
      RATIO_TO_REPORT(VIEWS) OVER() AS percentage,
      MAX(click_data.content_headline) AS headline,
      MAX(content_data.content) AS content
    FROM (
      SELECT
        content_id,
        content_headline,
        SUM(page_views) AS views
      FROM
          TABLE_DATE_RANGE([project_data.{}_data_],
                    TIMESTAMP('{}'),
                    TIMESTAMP('{}'))
      WHERE
        content_id IS NOT NULL 
        AND
        SECTION == '{}'
        AND
        content_headline IS NOT NULL
        AND
        TIMESTAMP(activity_date_time) BETWEEN TIMESTAMP('{}') AND TIMESTAMP('{}')
      GROUP BY
        content_id,
        content_headline) click_data
    INNER JOIN (
      SELECT
        content_id,
        content
      FROM
        [project_data.{}_content] ) content_data
    ON
      click_data.content_id = content_data.content_id
    GROUP BY
        content_id
    ORDER BY
        views DESC
    )) WHERE running < {}
    """

    path = 'article_contents_{}_{}_{}_{}/'.format(newspaper, section, start, end)

    if not os.path.exists(path):
        os.mkdir(path)

    query = query.format(newspaper, start_date, end_date, section, start, end, newspaper, str(percentage))


    views_df = pd.io.gbq.read_gbq(query=query, project_id=project_id)
        
    views_df['headline'] = views_df['headline']
    views_df['content'] = views_df['content']
        
        # select the articles that contribute to the most page views (e.g. top 20%)
        ### TO DO
        # save top articles into text files

    for [_, article] in views_df.iterrows():
        file_name = path + article['content_id'] + '.txt'
        html_content = article['content']
        html_headline = article['headline']

        soup_content = bs(html_content, "lxml")
        text_content = soup_content.get_text()

        soup_headline = bs(html_headline, "lxml")
        text_headline = soup_headline.get_text()

        with open(file_name, 'w') as f:
            f.write(text_headline)
            f.write('\n\n')
            f.write(text_content)
        
    return views_df


def get_articles_contents(newspaper, start, end, percentage=0.9):
    """
    Returns the ID, headline and content of all articles viewed during an anomaly.
    Creates a directory article_contents_total_start_end/ 
    and save each file in a txt.

    Input parameters:
    newspaper = "sun" or "times"
    section = the section to return articles from (? what about the ALL section...)
    start, end = start and end datetime of anomaly (strings formatted as 'YYYY-MM-DD HH:MM:SS')
    percentage = only gets the articles that represent more than 90% of the views.
    """

    project_id = "newsuk-datatech-s2ds-2017-2"

    # Start calculates the date.
    start_date = start.split()[0]
    start_label = start.replace(":", "-")
    start_label = start_label.replace(" ", "_")

    end_date = end.split()[0]
    end_label = end.replace(":", "-")
    end_label = end_label.replace(" ", "_")

    query = """
    SELECT content_id, headline, content, views
    FROM (
    SELECT content_id, percentage, SUM(percentage)
        OVER 
        (ORDER BY percentage DESC) as running,
        headline, content, views
    FROM (
    SELECT
      click_data.content_id AS content_id,
      MAX(click_data.views) AS views,
      RATIO_TO_REPORT(VIEWS) OVER() AS percentage,
      MAX(click_data.content_headline) AS headline,
      MAX(content_data.content) AS content
    FROM (
      SELECT
        content_id,
        content_headline,
        SUM(page_views) AS views
      FROM
          TABLE_DATE_RANGE([project_data.{}_data_],
                    TIMESTAMP('{}'),
                    TIMESTAMP('{}'))
      WHERE
        content_id IS NOT NULL 
        AND
        content_headline IS NOT NULL
        AND
        TIMESTAMP(activity_date_time) BETWEEN TIMESTAMP('{}') AND TIMESTAMP('{}')
      GROUP BY
        content_id,
        content_headline) click_data
    INNER JOIN (
      SELECT
        content_id,
        content
      FROM
        [project_data.{}_content] ) content_data
    ON
      click_data.content_id = content_data.content_id
    GROUP BY
        content_id
    ORDER BY
        views DESC
    )) LIMIT 100
    """

    path = 'article_contents_{}_{}_{}/'.format(newspaper, start, end)

    if not os.path.exists(path):
        os.mkdir(path)

    query = query.format(newspaper, start_date, end_date, start, end, newspaper, str(percentage))

    views_df = pd.io.gbq.read_gbq(query=query, project_id=project_id)

    views_df['headline'] = views_df['headline']
    views_df['content'] = views_df['content']

    # select the articles that contribute to the most page views (e.g. top 20%)
    ### TO DO
    # save top articles into text files
    
    clean_contents = []
    clean_headlines = []
    
    for [_, article] in views_df.iterrows():
        file_name = path + article['content_id'] + '.txt'
        html_content = article['content']
        html_headline = article['headline']
        
        print('heeey')

        soup_content = bs(html_content, "lxml")
        text_content = soup_content.get_text()
        clean_contents += [text_content]

        soup_headline = bs(html_headline, "lxml")
        text_headline = soup_headline.get_text()
        clean_headlines += [text_headline]

        with open(file_name, 'w') as f:
            f.write(text_headline)
            f.write('\n\n')
            f.write(text_content)

    views_df['content'] = np.array(clean_contents)
    views_df['headline'] = np.array(clean_headlines)
    
    return views_df

