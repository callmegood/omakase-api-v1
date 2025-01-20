"""
Created: 2024-12-16
Last Updated: 2024-12-18

Description:
- 
- 
"""


###########
# IMPORTS #
###########

import sys, time, re, json
import numpy as np
import pandas as pd
import asyncio
import nest_asyncio

from dotenv import load_dotenv
from openai import OpenAI
from getopt import getopt, GetoptError as opt_err
from functools import partial

from helper_functions import llm
from constants import *



#############
# FUNCTIONS #
#############

def is_topic_subtopic_level(row, topic, subtopic, level):
    topic = topic.strip().lower()
    subtopic = subtopic.strip().lower()
    level = level.strip().lower()

    topic_test = row[LLM_TOPIC].strip().lower()
    top_sub_test = row[LLM_SUBTOPIC].strip().lower()
    level_test = row[LLM_LVL].strip().lower()

    if pd.isnull(topic_test) or pd.isnull(top_sub_test) or pd.isnull(level_test):
        return False

    if re.search(topic, topic_test):
        top_sub_test = json.loads(top_sub_test)
        if subtopic in top_sub_test[topic]:
            if level == level_test:
                return True
    else:
        return False


def collate_phase_content(topic, subtopic, level, format, n_book=10, n_program=5):
    '''
     - topic, subtopic & level will be strictly used - will return < n items if only so many can be found 
     - format will be loosely used - will turn to other options to make n items
        - format - if programs cannot be found, will turn to books
     - Availability Check (not implemented at PoC, requires API to Cloud-ILS and EventBrite)
        - Programs must still be available at datetime of run
        - Book loan status and location at datetime of run to be gathered and displayed to patron
    '''
    df = pd.read_csv(CONTENT_FILE, dtype=object)
    df[PUB_DTE] = pd.to_datetime(df[PUB_DTE], format='%Y-%m-%d', errors='coerce')
    df.sort_values(by=[PUB_DTE], ascending=False, inplace=True, ignore_index=True)

    filter1 = 'is_topic_subtopic_level'
    df[filter1] = df.apply(partial(is_topic_subtopic_level, topic=topic, subtopic=subtopic, level=level), axis=1)
    df = df[df[filter1] == True]
    df.drop(filter1, axis=1, inplace=True)
 
    books = df[df[FORMAT].isin(['books', 'ebooks'])]
    if len(books) > n_book:
        books = pd.concat(
            [
                books[:n_book//2], 
                books[n_book//2:].sample(n_book-n_book//2)
            ], 
            axis=0, ignore_index=True
        )
    
    progs = df[df[FORMAT].isin(['programs'])]
    if len(progs) > n_program:
        progs = progs.sample(n_program)
    
    content = pd.concat([books, progs], axis=0, ignore_index=True)

    selected_content = content[content[FORMAT]==format]
    if len(selected_content) == 0:
        return content
    else:
        selected_content.reset_index(drop=True, inplace=True)
        return selected_content


def evaluate_omakases(omakases, topic, subtopic, overall_level):
    pass


def generate_selection_reason(oma, reason, topic, subtopic, overall_level):
    pass


def construct_output(oma, reason):
    output = dict()
    for i in range(len(PHASES)):
        output[PHASES[i]] = json.loads(oma[i].to_json())
    
    output["Reason"] = reason
    
    return json.dumps(output)


def generate_omakase(json_in, n_options=3, verbose=False):
    n_tokens = 0
    inputs = json.loads(json_in)
    
    topic = inputs["topic"]
    subtopic = inputs["subtopic"]
    overall_level = inputs["level"]

    phase_contents = dict()
    for phase in range(len(PHASES)):
        level = OMAKASES[overall_level][phase]["Level"]
        format = OMAKASES[overall_level][phase]["Format"]
        
        dfc = collate_phase_content(topic, subtopic, level, format)
        phase_contents.append(dfc)

    omakases = dict()
    for i in range(n_options):
        
        oma = []
        for phase in range(len(PHASES)):
            dfc = phase_contents[phase]
            if len(dfc) > 0:
                oma.append(dfc.sample(1).iloc[0])
            else:
                oma.append(pd.Series())
        
        omakases[f"Option {i}"] = oma

    oma, reason, n_tokens1 = evaluate_omakases(omakases, topic, subtopic, overall_level)
    final_reason, n_tokens2 = generate_selection_reason(oma, reason, topic, subtopic, overall_level)

    n_tokens += n_tokens1 + n_tokens2
    if verbose: print(f"Total tokens consumed = {n_tokens}")

    json_out = construct_output(oma, final_reason)
    return json_out
