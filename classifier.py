"""
Created: 2024-12-10
Last Updated: 2024-12-16

Description:
- Holds the functions to classify the content (books, programs/events)
- Chain-of-Thought (COT) technique is used explicity for the functions that classify topics, subtopics, level
  The reasonings for these are also stored for troubleshooting purposes
"""

###########
# IMPORTS #
###########

import sys, time, re, json
import numpy as np
import pandas as pd
import asyncio
import nest_asyncio
import datetime

from dotenv import load_dotenv
from openai import OpenAI
from getopt import getopt, GetoptError as opt_err

from helper_functions import llm
from constants import *



#############
# FUNCTIONS #
#############

def clean(input):
    '''If input is a string, strip. If input is NA, output as empty string'''
    output = "" if pd.isna(input) else str(input).strip()
    # output = input.strip() if input is not np.nan else "" 
    return output


def clean_date(sdate):
    if sdate is not np.nan:
        sdate = sdate.strip()
    else:
        return np.nan
    
    if re.search(r'^\d{4}-\d{2}-\d{2}$', sdate):            # Desired format
        pass

    elif re.search(r'^\d{2}/\d{2}/\d{4}$', sdate):           # MM\DD\YYYY
        t1 = time.strptime(sdate, '%m/%d/%Y')
        sdate = time.strftime('%Y-%m-%d', t1)
    
    elif re.search(r'^\d{4}-\d{2}$', sdate):                # YYYY-MM only
        t1 = time.strptime(sdate, '%Y-%m')
        sdate = time.strftime('%Y-%m-%d', t1)

    elif re.search(r'20\d{2}|19\d{2}', sdate):                 # Presence of YYYY (19xx, 20xx) only
        sdate = re.findall(r'20\d{2}|19\d{2}', sdate)[0] + '-01-01'
    
    else:
        sdate = np.nan

    return sdate


def derive_pub_date(df):
    '''
    Derive a consistent publication date using (i) DWH's TITLE_PUB_DTE; (ii) Google API's G_PUB_DTE
    Logic:
     - Between TITLE_PUB_DTE & G_PUB_DTE, use the one with latest date
       Reason is that some books might be re-published, and such books indicate 'freshness'
       The Omakase's intent is to recommend 'fresh' up-to-date content
    '''
    def get_larger_date(row):
        date1 = row[tcol1]
        date2 = row[tcol2]

        if pd.isnull(date1) and pd.isnull(date2):
            return pd.NaT  # Both dates are NaT
        elif pd.isnull(date1):
            return date2
        elif pd.isnull(date2):
            return date1
        else:
            return max(date1, date2)

    tcol1 = 'temp_date1'
    tcol2 = 'temp_date2'
    df[tcol1] = pd.to_datetime(df[TITLE_PUB_DTE].apply(clean_date), format='%Y-%m-%d', errors='coerce')
    df[tcol2] = pd.to_datetime(df[G_PUB_DTE].apply(clean_date), format='%Y-%m-%d', errors='coerce')
    
    df[PUB_DTE] = df.apply(get_larger_date, axis=1)

    df.drop([tcol1, tcol2], axis=1, inplace=True)

    return df


def clean_media_format(df):
    '''
    Convert MEDIA values into standardised values
    Only applies to book content data
    '''
    def convert_media_format(row):
        format = row[MEDIA]
        if format is not np.nan:
            format = format.strip().lower()
        else:
            return np.nan
        
        if re.search(r'ebook', format):
            return 'ebook'
        elif re.search('book', format):
            return 'book'

    df[FORMAT] = df.apply(convert_media_format, axis=1)
    return df


async def extract_page_count(row, model=MODEL_DEFAULT, verbose=False):
    '''
    Function to derive page count from N_PHY_DESC
    Output should be either integer or NA
    '''
    n_tokens = 0

    phy_desc = clean(row[N_PHY_DESC])

    prompt = f"""
    Given the phsyical description, extract the page count: 
    {DELIM}{phy_desc}{DELIM}

    Output MUST be as an integer
    OR
    If there is insufficient info for page count to be extracted, output MUST be empty string
    """

    response = await llm.get_completion(prompt, model=model, verbose=verbose)    
    messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    n_tokens += llm.num_tokens_from_messages(messages, model)

    try:
        response = str(int(response))
    except:
        response = np.nan
    
    return response, n_tokens


async def interpret_DDC(row, model=MODEL_DEFAULT, verbose=False):
    '''
    Function to interpret the DDC code as a text description
    If no viable response, output as NA
    '''
    n_tokens = 0

    ddc = clean(row[DDC])

    prompt = f"""
    Give a short interpretation of the following Dewey Decimal Code (DDC): 
    {DELIM}{ddc}{DELIM}

    Avoid explaining the breakdown of the DDC.
    If no DDC was given or it cannot be interpreted, output MUST be empty string
    """
    
    response = await llm.get_completion(prompt, model=model, verbose=verbose)
    messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    n_tokens += llm.num_tokens_from_messages(messages, model)

    if len(response.strip()) < 5: 
        response = np.nan

    return response, n_tokens


async def classify_topic(row, model=MODEL_DEFAULT, verbose=False):
    '''
    Function to classify topic from content's metadata:
     - TITLE, N_ABST, N_SUBJ/G_SUBJ, LLM_DDC
    Limited to classes defined in TOPICS
    Output as json list. If no viable response, output as NA
    '''
    delim = '###'
    n_tokens = 0

    subjects = clean(row[N_SUBJ])
    if subjects == '': subjects = clean(row[G_SUBJ])

    prompt = f"""
    Given the content (book) information:
    <title>{clean(row[TITLE])}</title>
    <blurb>{clean(row[N_ABST])}</blurb>
    <subjects>{subjects}</subjects>
    <dewey_decimal_summary>{clean(row[LLM_DDC])}</dewey_decimal_summary>
    Classify this content using this list of topics: {TOPICS}
    For definition of each topic, take reference from its associated subtopics: {TOPICS_SUBTOPICS}
    
    Your response MUST be in the form of a json list.
    The content can be classified using more than 1 topic.
    If none of the topics above applies to the content, output an empty list.
    State your reason after the json list, separated by {delim}
    """
    
    llm_response = await llm.get_completion(prompt, model=model, verbose=verbose)
    messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': llm_response}]
    n_tokens += llm.num_tokens_from_messages(messages, model)

    # Cleaning the LLM output
    if verbose: print(f"LLM raw output for Topic: \n{llm_response}\n")
    response = llm_response.split(delim)[0].strip().title()
    try: 
        reason = llm_response.split(delim)[1].strip()
    except: reason = ""

    response = re.sub(r'\n', ' ', response)
    # response = re.findall(r'\[[ &"\-\',a-zA-Z]*\]', response)[0]  # r"\[.*\]"
    try:
        response = re.findall(r'\[[ &"\-\',a-zA-Z]*\]', response)[0]  # r"\[.*\]"
        response = ','.join(eval(response)) # llm_ts = ','.join(json.loads(llm_ts))
    except:
        print(f"Topic output from LLM supposed to be a json-list but output as {response}") 

    response = np.nan if response == '' else response

    return response, reason, n_tokens


async def classify_subtopic(row, model=MODEL_DEFAULT, verbose=False):
    '''
    Function to classify sub-topics for each derived topic derived by classify_topic()
    Metadata used:
     - TITLE, N_ABST, N_SUBJ/G_SUBJ, LLM_DDC, LLM_TOPIC
    For each topic already derived, classes are limited to those defined in TOPICS_SUBTOPICS
    Output as json. If no viable response, output as NA
    '''
    delim = '###'
    n_tokens = 0
    reason = ''

    response_dict = dict()

    topics_record = clean(row[LLM_TOPIC])
    subjects = clean(row[N_SUBJ])
    if subjects == '': subjects = clean(row[G_SUBJ])

    for topic in TOPICS:
        if topic in topics_record:
            sub_topics = TOPICS_SUBTOPICS[topic]
        else:
            continue

        prompt = f"""
        Given the content (book) information:
        <title>{clean(row[TITLE])}</title>
        <blurb>{clean(row[N_ABST])}</blurb>
        <subjects>{subjects}</subjects>
        <dewey_decimal_summary>{clean(row[LLM_DDC])}</dewey_decimal_summary>
        This book was already classified as topic: {topic}
        Using the content information above, classify it using this list of sub-topics that is related to the topic: {sub_topics}
        
        Your response MUST be in the form of a json list.
        The content can be classified using more than 1 sub-topic.
        If none of the sub-topics above applies to the content, output an empty list.
        State your reason after the json list, separated by {delim}
        """    

        llm_response = await llm.get_completion(prompt, model=model, verbose=verbose)
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': llm_response}]
        n_tokens += llm.num_tokens_from_messages(messages, model)

        # Cleaning the LLM output
        if verbose: print(f"LLM raw output for topic `{topic}` - its subtopic: \n{llm_response}\n")
        response = llm_response.split(delim)[0].strip().title()
        try: 
            reason += '*' + llm_response.split(delim)[1].strip() + '\n'
        except:
            reason += '*' + "" + '\n'

        response = re.sub(r'\n', ' ', response)
        response = re.findall(r'\[[ &"\-\',a-zA-Z]*\]', response)[0]  # r"\[.*\]"
        
        try:
            response_dict[topic] = json.loads(response)
            # response = ','.join(eval(response))
        except:
            print(f"Topic-Subtopic output from LLM supposed to be json but for topic {topic}, LLM output as {response}") 

    if len(response_dict) > 0:
        return json.dumps(response_dict), reason, n_tokens
    else:
        return '', reason, n_tokens


async def classify_level(row, model=MODEL_DEFAULT, verbose=False):
    '''
    Function to classify the level of the topic using metadata:
     - TITLE, N_ABST, N_SUBJ, G_SUBJ, LLM_DDC
     - If available: LLM_PG_CNT, LLM_SUBTOPICS
    The core of the prompt lies in the definitions for Beginner, Intermediate, Advanced
    Output as one of the strings {Beginner, Intermediate, Advanced, Unknown} 
    '''
    delim = '###'
    n_tokens = 0

    subjects = clean(row[N_SUBJ])
    if subjects == '': subjects = clean(row[G_SUBJ])

    prompt_part1 = f"""
    Given the content (book) information:
    <title>{clean(row[TITLE])}</title>
    <blurb>{clean(row[N_ABST])}</blurb>
    <subjects>{subjects}</subjects>
    <dewey_decimal_summary>{clean(row[LLM_DDC])}</dewey_decimal_summary>
    """

    # If page count is available as additional info
    # print(row[LLM_PG_CNT])
    pg_cnt = clean(row[LLM_PG_CNT])
    prompt_pg_cnt = ''
    if pg_cnt != '':
        prompt_pg_cnt = f"""
        <page_count>book has {pg_cnt} pages</page_count>
        """

    # If topic and subtopics are available as additional info
    topic_subtopics = clean(row[LLM_SUBTOPIC])
    prompt_topic_subtopic = ''
    if topic_subtopics != '':
        prompt_topic_subtopic = f"""
        This book was classified with the following topics and associated subtopics: {topic_subtopics}

        """

    prompt_part2 = f"""
    Inferring from the information above, classify the level of the content as either:
     * Beginner - suitable for someone who just started out on the topic or has superficial knowledge
     * Intermediate - suitable for someone who has broad knowledge of the topic and desires more depth or specialisation
     * Advanced - suitable for someone who wants pure depth and specialisation
    
    Your response MUST be a single word, one of: {LEVELS}
    The content CANNOT be classified using more than 1 level.
    If unable to infer the topic level, please return as "{UNK}"
    State your reason after the output, separated by {delim}
    """    

    prompt = prompt_part1 + prompt_topic_subtopic + prompt_part2
    llm_response = await llm.get_completion(prompt, model=model, verbose=verbose)
    messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': llm_response}]
    n_tokens += llm.num_tokens_from_messages(messages, model)

    if verbose: print(f"LLM raw output for Level: \n{llm_response}\n")
    response = llm_response.split(delim)[0].strip().title()
    response = response.strip('"')
    try:
        assert response in LEVELS + [UNK.title()], f"Incorrect derived level <{response}> for ISBN <{row[ISBN]}> title <{row[TITLE]}>"
    except: 
        response = ""
    try:
        reason = llm_response.split(delim)[1].strip()
    except: reason = ""
    return response, reason, n_tokens


def async_classifier_wrapper(df, col, verbose=False):
    '''
    Wrapper function that assist in controlling of:
     (i)  rows to execute
     (ii) columns to execute
    '''
    len_df = len(df)

    if col == LLM_TOPIC:
        classifier_fx = classify_topic
        
        if col in df.columns:
            to_fill = df[col].isna() 
            df1 = df[to_fill]
            df2 = df[~to_fill] # Only for empty topic then call LLM, if have already don't need call
        else:
            df1 = df
            df2 = pd.DataFrame()
    
    elif col == LLM_LVL:
        classifier_fx = classify_level

        # Only do this for those with topics classified, so LLM_TOPIC must exist!
        if col in df.columns:
            to_fill = df[col].isna() & ~df[LLM_TOPIC].isna()
        else:
            to_fill = ~df[LLM_TOPIC].isna()
        df1 = df[to_fill]
        df2 = df[~to_fill]
    
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(classifier_fx(row, model=MODEL_DEFAULT, verbose=verbose)) 
        for _, row in df1.iterrows()
    ]
    df1[[col, col+' - '+REASON, col + ' - '+N_TOKENS]] = loop.run_until_complete(asyncio.gather(*tasks)) 

    # For classification of topic, need to classify subtopic as a natural next step
    # But only do this for those with topics assigned
    if col == LLM_TOPIC:
        to_fill2 = ~df1[col].isna() 
        df11 = df1[to_fill2]
        # print(df11)
        df12 = df1[~to_fill2]

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        tasks = [
            loop.create_task(classify_subtopic(row, model=MODEL_DEFAULT)) 
            for _, row in df11.iterrows()
        ] 
        df11[[LLM_SUBTOPIC, LLM_SUBTOPIC+' - '+REASON, LLM_SUBTOPIC + ' - ' + N_TOKENS]] = loop.run_until_complete(asyncio.gather(*tasks))

        df1 = pd.concat([df11, df12], axis=0, ignore_index=True)

    df = pd.concat([df1, df2], axis=0, ignore_index=True)

    assert len_df == len(df)   # Check num of rows before and after
    return df


def derive_supporting_data(df=None):
    '''
    Function to derive LLM_DDC, LLM_PG_CNT from the relevant metadata fields
    '''
    if df is None:
        df = pd.read_csv(CONTENT_FILE, dtype=object)

    supp_cols = [LLM_DDC, LLM_PG_CNT]
    derive_fxs = [interpret_DDC, extract_page_count]

    for i, col in enumerate(supp_cols):
        if col in df.columns:
            to_fill = df[col].isna() 
            df1 = df[to_fill]
            df2 = df[~to_fill]
        else:
            df1 = df
            df2 = pd.DataFrame()

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        tasks = [
            loop.create_task(derive_fxs[i](row, model=MODEL_DEFAULT)) 
            for _, row in df1.iterrows()
        ]
        df1[[col, col + ' - ' + N_TOKENS]] = loop.run_until_complete(asyncio.gather(*tasks))

        df = pd.concat([df1, df2], axis=0, ignore_index=True)

    return df


def classify_content(
    df=None,
    do_missing_only=True, do_all = True,
    do_topic = True, do_level = True,
    derive_supporting = True,
    verbose = False
):
    '''
    Main function that reads in Content's metadata file and with LLM help, derive the required metadata:
     - Step 0. Clean and derive Publication Date using source from NLB DWH & Google API
     - Step 1. {LLM_DDC, LLM_PG_CNT}
     - Step 2. {LLM_TOPIC, LLM_SUBTOPIC, LLM_LVL}
    Function allows control of:
     (i)  rows to execute via "do_missing_only". 
          - By default, only do those rows with missing values.
          - When set as do_missing_only=False and do_all=True, we remove all existing LLM_XX columns and re-do everything
     (ii) columns to execute via "do_all", "do_topic", "do_level"
    '''
    if df is None:
        df = pd.read_csv(CONTENT_FILE, dtype=object)
    
    df = derive_pub_date(df)
    df = clean_media_format(df)

    llm_cols = [LLM_TOPIC, LLM_LVL]
    supp_cols = [LLM_DDC, LLM_PG_CNT]

    if derive_supporting or (not set(supp_cols) < set(df.columns)):
        df = derive_supporting_data(df)

    if do_all and not do_missing_only: # remove all existing llm columns and re-classify
        for col in llm_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

    if do_all: # the classify_xxx() functions will by default classify rows with missing entries
        for col in llm_cols:
            df = async_classifier_wrapper(df, col, verbose)
    
    else: # classify individual requested columns
        if do_topic: # topic & sub-topic has to be classified together
            df = async_classifier_wrapper(df, LLM_TOPIC, verbose)
        
        if do_level:
            df = async_classifier_wrapper(df, LLM_LVL, verbose)

    llm.close_llm_records()

    return df



if __name__ == "__main__":
    CONTENT_FILE = './data/dig_for_class.csv'
    DF = pd.read_csv(CONTENT_FILE, sep = "|", on_bad_lines= 'skip')
    # Define total number of calls (number of library to ADZ combinations)
    total_calls = DF.shape[0]

    # Define rate limit 
    rate_limit = 500
    calls_made = 1000

    # Calculate the number of intervals needed to make all the calls (1481)
    intervals = total_calls // rate_limit
    remaining_calls = total_calls % rate_limit

    # list to store timing

    for i in range(intervals):
        print(calls_made)
        print(calls_made + rate_limit)
        df = DF.iloc[range(calls_made, calls_made + rate_limit), :]
        calls_made += rate_limit
        # df = derive_supporting_data(df)
        df = classify_content(df)
        OUTPUT_PATH = f"./data/zz_archive/DIG_processed_interval_{i}.csv"
        df.to_csv(OUTPUT_PATH, index=False)
        
    df = DF.iloc[range(calls_made, calls_made + remaining_calls), :]
    calls_made += remaining_calls 
    df = classify_content(df)
    OUTPUT_PATH = f"./data/zz_archive/DIG_processed_interval_final.csv"
    df.to_csv(OUTPUT_PATH , index=False)
    
    # ---------------------------------------------------
    # Testing
    # ---------------------------------------------------

    