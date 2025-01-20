"""
Created: 2024-12-16
Last Updated: 2025-01-15
Last Updated By: Des

Description:
- Taking user inputs for topic, subtopic and level, 
- collate_phase_content(): 
    For each phase extract relevant content from CONTENT_FILE, subset to 5 latest titles and 5 random samples for discovery
- Create 3 omakase set options by random sampling 1 from subset for each phase.
- evaluate_omakase(): 
    Send 3 sets to LLM to get it to evaluate and return best option

For discussion:
- Edge case: What if there is only 1 book for the whole level, 
e.g. only 1 advanced book - do we return that 1, do we not return?
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

def clean(input):
    '''If input is a string, strip. If input is NA, output as empty string'''
    output = input.strip() if input is not np.nan else "" 
    return output


def is_topic_subtopic_level(row, topic, subtopic, level): # checking whether topic and subtopic combinations are valid
    '''
    Function to check if a content record (row) satisfies the input definitions (topic, subtopic, level)
    If satisfies all 3 definitions, return True, else return False
    '''
    topic = topic.strip().lower() # provided/posted by user interface
    subtopic = subtopic.strip().lower()
    level = level.strip().lower()

    topic_test = clean(row[LLM_TOPIC]).strip().lower() # in content database
    top_sub_test = clean(row[LLM_SUBTOPIC]).strip().lower()
    level_test = clean(row[LLM_LVL]).strip().lower()

    if pd.isnull(topic_test) or pd.isnull(top_sub_test) or pd.isnull(level_test): # make sure no data is missing
        return False

    if re.search(topic, topic_test): # if can find topic in row of database
        top_sub_test = json.loads(top_sub_test)
        if subtopic in top_sub_test[topic]:
            if level == level_test:
                return True
    else:
        return False


def collate_phase_content(topic, subtopic, level, format, n_book=10, n_program=5, fill_with_lower_lvl=False):
    '''
    Given the topic, subtopic, level & format inputs, return the selected content records,
    which will be used for a single phase/course in the omakase 3-course journey generation

    Inputs
     - Definition parameters: topic, subtopic, level, format
     - n_book controls the maximum book content records to select
     - n_program controls the maximum program content records to select
     - fill_with_lower_lvl determines if we allow next lower level (eg, Beginner) content 
       to fill any shortfall with current level (eg, Intermediate)
     - format will determine the final selection of content records to return

    Notes
     - Availability Check (not implemented at PoC, requires API to Cloud-ILS and EventBrite)
        - Programs must still be available at datetime of run
        - Book loan status and location at datetime of run to be gathered and displayed to patron
    '''
    # Input overall content database, sort them by published dates
    # Content should contain books, ebooks, programs/events
    df = pd.read_csv(CONTENT_FILE, dtype=object)
    df[PUB_DTE] = pd.to_datetime(df[PUB_DTE], format='%Y-%m-%d', errors='coerce')
    df.sort_values(by=[PUB_DTE], ascending=False, inplace=True, ignore_index=True)

    # Filter only the content records that meet the topic, subtopic, level
    filter1 = 'is_topic_subtopic_level'
    df[filter1] = df.apply(partial(is_topic_subtopic_level, topic=topic, subtopic=subtopic, level=level), axis=1)
    df1 = df[df[filter1] == True]
    df.drop(filter1, axis=1, inplace=True) # drop column = "is_topic_subtopic_level"

    # Filter content records that meet the topic, subtopic, level-1 (for purpose of shortfall-filler)
    level_low = LEVELS[0] # default is Beginner
    df2 = pd.DataFrame(columns=df.columns)
    for i, lvl in enumerate(LEVELS):
        if lvl == level and i-1 >= 0: 
            level_low = LEVELS[i-1] 
    if level_low != LEVELS[0]:
        df[filter1] = df.apply(partial(is_topic_subtopic_level, topic=topic, subtopic=subtopic, level=level_low), axis=1)
        df2 = df[df[filter1] == True]
        df.drop(filter1, axis=1, inplace=True) # drop column = "is_topic_subtopic_level"

    # For book content, select n_book rows, half is by latest published date, half is random select 
    # If initial book content count less than n_book, either
    #   Return the whole list or
    #   Fill the shortfall with next lower level book content (randomly selected)
    books = df1[df1[FORMAT].isin(['book', 'ebook'])]
    books_low = df2[df2[FORMAT].isin(['book', 'ebook'])]
    if len(books) > n_book: # 
        books = pd.concat(
            [
                books[:n_book//2],                          # selection by published date order
                books[n_book//2:].sample(n_book-n_book//2)  # random selection
            ], 
            axis=0, ignore_index=True
        )
    elif len(books) < n_book and fill_with_lower_lvl:
        n_remain = n_book - len(books)
        if len(books_low) >= n_remain:
            books = pd.concat([books, books_low.sample(n_remain)], axis=0, ignore_index=True)
    else:
        pass
    
    # For program content, select n_program rows, all of which by random selection
    # If initial program content count less than n_program, just return the whole list 
    progs = df1[df1[FORMAT].isin(['programs'])]
    if len(progs) > n_program:
        progs = progs.sample(n_program)
    
    # Combine the book & program content
    content = pd.concat([books, progs], axis=0, ignore_index=True)

    # Example: if phase = "Appetiser", format = 'programs', then selected_content should contain programmes,
    # but if it is empty, return the combined content (which should contain only book content)
    ls_format = ['book', 'ebook'] if format == 'book' else [format]
    selected_content = content[content[FORMAT].isin(ls_format)]
    if len(selected_content) == 0:
        return content
    else:
        selected_content.reset_index(drop=True, inplace=True)
        return selected_content


def subset_content_columns(item):
    '''Given item (pd.Series object), return only the relevant attributes/index'''
    if item.empty:
        return item
    else:
        item_subset = item[[TITLE, 
                            AUTH, 
                            N_ABST, 
                            N_PHY_DESC, 
                            DDC, 
                            LLM_DDC, 
                            PUB_DTE, 
                            LLM_TOPIC, 
                            LLM_SUBTOPIC, 
                            LLM_LVL, 
                            FORMAT]]
        return item_subset


def evaluate_omakases(omakases, topic, subtopic, overall_level, n_options, model=MODEL_DEFAULT, verbose=False):
    '''
    Given n_options worth of omakase journeys - stored in dict omakases, enlist LLM to evaluate the options,
    and decide which option is the best, considering the topic, subtopic & overall_level

    Return the index representing the option, and the reasoning for the selection
    '''
    n_tokens = 0

    # parse json output, prep data to be fed to prompt / open AI, just need option, title, abstract llm topic subtopic and level reasons. 
    # For each content data in the omakase journeys, extract only the relevant metadata 
    omakases_subset = dict()
    for option, omakase in omakases.items():  # omakase is a list of pd.Series objects, each pd.Series holds the content metada
        omakases_subset[option] = [subset_content_columns(content) for content in omakase]
    
    # Prompts
    system_message = f"""
    Imagine you are a {overall_level} learner, interested in the general topic {topic} and more specifically {subtopic}.\
    You are considering which curated set of content will best suit your learning needs. You are impartial to books one level \
    above your level, for example, if you are a Beginner, you are willing to read Intermediate books, and if you are an Intermediate,\
    you are willing to read Advanced books. 
    """

    prompt= f"""
    Step 1: Consider the following curated sets of books or programmes below, where each set is an option, and provide bullet points of\
    the good and bad areas of each option. 
    <options>{omakases_subset}</options>

    Step 2: Given the pros and cons specified in Step 1, decide on the best option. Return ONLY a single digit number \
    between 0 to {n_options-1} denoting the best option after triple backtickes {DELIM}. \
    Return 0 if there is only a single option available.
    """

    # Generate response to get final recommended omakase and the reason it was selected
    messages = [{'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}]
    response = asyncio.run(llm.get_completion_from_messages(messages, model=model, verbose=verbose))    

    try: 
        reason = response.split(DELIM)[0].strip()
        oma = response.split(DELIM)[1].strip()
    except:
        print("unable to extract selected option based on LLM's output")
        print(response)
    messages.append({'role': 'assistant', 'content': response})
    n_tokens += llm.num_tokens_from_messages(messages, model)
    
    return oma, reason, n_tokens


def generate_selection_reason(oma, reason, topic, subtopic, overall_level, model=MODEL_DEFAULT, verbose=False):
    n_tokens = 0
    
    prompt = f"""
    You are a knowledgeable library who curates sets of content for your patrons based on their requested \
    {topic}, {subtopic} and {overall_level}. You have decided on a set of content, denoted by <content> for this particular patron,\
    based on the reasons denoted in <reason> below. 

    <content> {oma} </content>
    <reason> {reason} </reason>

    Read and understand the <content> and <reason>, then sell this learning package to your patron in less than 100 words. \
    Your output should only be the 100 word explanation.
    """
    
    response = asyncio.run(llm.get_completion(prompt, model=model, verbose=verbose)) 
    messages = [{'role': 'user', 'content': prompt}]
    n_tokens += llm.num_tokens_from_messages(messages, model)
    
    return response, n_tokens


def construct_output(oma, reason):
    '''
    Input
    - 1. oma - list of qty 3 pd.Series, each holding the metadata of the content
    - 2. reason - the generated reason for selecting the final omakase, which will be displayed to the user
    Output
    - json text converted from dictionary with keys: Appetiser, Main, Dessert, Reason 
    '''
    output = dict()
    for i in range(len(PHASES)):
        output[PHASES[i]] = json.loads(oma[i].to_json())
    
    output["Reason"] = reason
    
    return json.dumps(output)


def generate_omakase(topic, subtopic, overall_level, n_options=3, verbose=False): # orchestrator, json_in is from application
    '''
    Main orchestrator function that generates the final omakase (Appetiser, Main, Dessert) with a reason
    for displaying to user
    
    Overall algorithm:
    - Takes in user inputs: topic, subtopic, overall level
      These will be used to define the individual oma course's level and content format as defined in OMAKASES
    - For each oma course, using the specific level & format, we collate the possible contents
    - Use the collated contents for the courses, generate n_options permutated omakases
    - Evaluate the n_options and select the final one for output
    - Generate the reason for final selection, which will be displayed to user
    '''
    n_tokens = 0
    # inputs = json.loads(json_in)
    
    # topic = inputs["topic"]
    # subtopic = inputs["subtopic"]
    # overall_level = inputs["level"]

    phase_contents = []
    for phase in range(len(PHASES)): # will collect options for all three phases
        level = OMAKASES[overall_level][phase]["Level"]
        format = OMAKASES[overall_level][phase]["Format"]
        
        # Only allow next lower level content to fill shortfall in current level for the scenarios:
        #  1. Dessert phase of Beginner or Intermediate overall level
        #  2. Advanced overall level
        fill_with_lower_lvl = False
        if (overall_level in LEVELS[:2] and phase == 2) or (overall_level == LEVELS[2]):
            fill_with_lower_lvl = True
        
        dfc = collate_phase_content(topic, subtopic, level, format, fill_with_lower_lvl=fill_with_lower_lvl) 
        phase_contents.append(dfc) # list of 3 dataframes each with either 5 programmes or 10 books
    
    omakases = dict()
    for i in range(n_options): # generate 3 different omakases out of the options from the above
        oma = []
        for phase in range(len(PHASES)):
            dfc = phase_contents[phase] 
            dfc = dfc[~dfc.isin(oma)] # newly added line to avoid duplicate recommendations in each omakase
            if len(dfc)>0: 
                oma.append(dfc.sample(1).iloc[0])
            else:
                oma.append(pd.Series())
        
        omakases[f"Option {i}"] = oma # generate one omakase option
    # End of loop, will have 3 different omakase options in dictionary (keys: Option 0, 1, ...)
    
    # Check that there is at least one viable omakase option - meaning it has at least 1 out of 3 contents:
    # Why check Option 0 only? 
    # Bcos if it is full empty (0 of 3 contents), remaining options will be full empty by current logic above
    if any(not s.empty for s in omakases.get("Option 0")):       
        llm_option, reason, n_tokens1 = evaluate_omakases(omakases, topic, subtopic, overall_level, n_options) #LLM to evaluate omakase
        oma = omakases[f"Option {llm_option}"]
        final_reason, n_tokens2 = generate_selection_reason(oma, reason, topic, subtopic, overall_level)

        n_tokens += n_tokens1 + n_tokens2
    else:
        oma = [pd.Series for i in range(len(PHASES))]
        final_reason = """Sorry, we could not find any content matching your selection criteria, \
            please change your selections and try again."""

    if verbose: print(f"Total tokens consumed = {n_tokens}")

    json_out = construct_output(oma, final_reason)
    return json_out



if __name__ == "__main__":
    json_in = """{
    "topic": "Personal Development",
    "subtopic": "Self-Help",
    "level": "Intermediate"
    }"""
    test_output = generate_omakase(json_in)
    with open("test_output.txt","w") as file:
        file.write(test_output)
