"""
Created: 2024-12-10
Last Updated: 2024-12-15

Description:
- Holds functions that are used to interact with the OpenAI API via the OpenAI.chat.completions.create API, 
  and to count tokens. 
  The 2 main functions are: get_completion() & get_completion_from_messages()
  These functions were adapted from GovTech's AI Champions BootCamp Run #1 in 2024

- In addition, there is a built in mechanism to store OpenAI API calls' responses to avoid duplicative calls (save $)
  - Each unique call is represented by 
    (i)  12 byte hash of a json that contains: messages, model, temperature, top_p, max_tokens, n
    (ii) LLM's response
  - Each unique call is saved into a repo file LLM_COMP_FILE
  - LLM_COMP_FILE is saved every 10 (can be adjusted) API calls
"""


###########
# IMPORTS #
###########

import os, json
import tiktoken
import pandas as pd
# import asyncio
# import nest_asyncio

from dotenv import load_dotenv
from openai import OpenAI
from getopt import getopt, GetoptError as opt_err
from cryptography.hazmat.primitives import hashes

from constants import MODEL_DEFAULT, LLM_COMP_FILE, HASH_STR, LLM_RAW



############
# MUST RUN #
############

load_dotenv('.env')

client = OpenAI()
SALT = os.getenv("SALT")

if not os.path.exists(LLM_COMP_FILE):
    DF_LLM = pd.DataFrame(columns=[HASH_STR, LLM_RAW])
else:
    DF_LLM = pd.read_csv(LLM_COMP_FILE, dtype=object)



#############
# FUNCTIONS #
#############

def hash(plaintext: str, encoding: str = 'utf-8') -> str:
    '''
    Function concatenates plaintext with salt, then hash it.
    The output is then truncated to 12 bytes and converted to hexadecimal string before being output.
     - 12 bytes --> space of 2**96 possibilities
    '''
    digest = hashes.Hash(hashes.SHA256())
    digest.update(bytes(plaintext + SALT, encoding=encoding))
    ciphertext = digest.finalize()[:12].hex()  # Truncate to first 12 bytes (96 bits) then convert to hex string (24 char)

    return ciphertext


def close_llm_records(drop_dup=True):
    '''Run this to save DF_LLM to LLM_COMPFILE'''
    if drop_dup:
        DF_LLM.drop_duplicates(subset=[HASH_STR], inplace=True)
    DF_LLM.to_csv(LLM_COMP_FILE, index=False)


# This is the "Updated" helper function for calling LLM
async def get_completion(prompt, model=MODEL_DEFAULT, temperature=0, top_p=1.0, max_tokens=1024, n=1, 
                         verbose=False, ignore_previous=False, n_save=10):
    messages = [{"role": "user", "content": prompt}]
    
    inputs = {
        "messages": messages, 
        "model":model, 
        "temperature":temperature, "top_p":top_p, 
        "max_tokens":max_tokens, "n":n
    }
    hash_str = hash(json.dumps(inputs))

    response_str = ""
    response_str_exist = False

    # Get LLM response from historical records first
    try:
        response_str = DF_LLM.loc[DF_LLM[HASH_STR] == hash_str, LLM_RAW].iloc[0]  # the .loc returns a series, so have to use .iloc[0] to retrieve the value
        response_str_exist = True
    except:
        if verbose: print(f"No historical record of inputs")

    # If LLM response not available, use OpenAI API to get response
    if ignore_previous or response_str == "" or DF_LLM is None:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            # response_format={ "type": "json_object" },
        )
        response_str = response.choices[0].message.content
    
    # Store LLM response if possible
    if DF_LLM is not None and response_str != "" and not response_str_exist:
        new_row = {HASH_STR: hash_str, LLM_RAW: response_str}
        DF_LLM.loc[len(DF_LLM)] = new_row
        if len(DF_LLM) % n_save == 0:  # save records every n_save LLM calls
            close_llm_records(drop_dup=False)
    elif response_str_exist and ignore_previous: # replace previous LLM output
        DF_LLM.loc[DF_LLM[HASH_STR] == hash_str, LLM_RAW] = response_str

    return response_str


# This a "modified" helper function that we will discuss in this session
# Note that this function directly take in "messages" as the parameter.
async def get_completion_from_messages(messages, model=MODEL_DEFAULT, temperature=0, top_p=1.0, max_tokens=1024, n=1, 
                                       verbose=False, ignore_previous=False, n_save=10):
    inputs = {
        "messages": messages, 
        "model":model, 
        "temperature":temperature, "top_p":top_p, 
        "max_tokens":max_tokens, "n":n
    }
    hash_str = hash(json.dumps(inputs))

    response_str = ""
    response_str_exist = False

    # Get LLM response from historical records first
    try:
        response_str = DF_LLM.loc[DF_LLM[HASH_STR] == hash_str, LLM_RAW].iloc[0]  # the .loc returns a series, so have to use .iloc[0] to retrieve the value
        response_str_exist = True
    except:
        if verbose: print(f"No historical record of inputs")
    
    # If LLM response not available, use OpenAI API to get response
    if ignore_previous or response_str == "" or DF_LLM is None:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            # response_format={ "type": "json_object" },
        )
        response_str = response.choices[0].message.content

    # Store LLM response if possible
    if DF_LLM is not None and response_str != "" and not response_str_exist:
        new_row = {HASH_STR: hash_str, LLM_RAW: response_str}
        DF_LLM.loc[len(DF_LLM)] = new_row
        if len(DF_LLM) % n_save == 0:  # save records every n_save LLM calls
            close_llm_records(drop_dup=False)
    elif response_str_exist and ignore_previous: # replace previous LLM output
        DF_LLM.loc[DF_LLM[HASH_STR] == hash_str, LLM_RAW] = response_str

    return response_str


# This function is for calculating the tokens given the "messages"
# ⚠️ This is simplified implementation that is good enough for a rough estimation
# For accurate estimation of the token counts, please refer to the "Extra" at the bottom of this notebook
def num_tokens_from_messages(messages, model=MODEL_DEFAULT):
    encoding = tiktoken.encoding_for_model(model)   # 'gpt-3.5-turbo', 'gpt-4o'
    value = ' '.join([msg.get('content') for msg in messages])
    return len(encoding.encode(value))


# Recommend to use this function for calculating the tokens in actual projects
# This is especially if the APIs involve multi-turns chat between the LLM and the users
# For more details, See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens

# Don't worry about understanding this function line-by-line, it's a utility tool
# The core function boils down to this: `encoding.encode(value)` in the last few lines of the code
def num_tokens_from_messages_accurate(messages, model="gpt-3.5-turbo"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106"
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-turbo",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613"]:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        # Old model: https://platform.openai.com/docs/deprecations
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message   # Account for the 3 special tokens: <start>, <end>, <role>.. refer below
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens