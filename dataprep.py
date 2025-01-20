from constants import *
import pandas as pd
import glob

# getting top 10000 most ranked titles.

def reorder_content_file(qsfile = "data/extract20241127/Extract 20241127_truncated_dig50k.csv",
                        api_output = "data/Output - Digital.csv"):
    dig = pd.read_csv(qsfile, dtype = object)
    dig_api = pd.read_csv(api_output, on_bad_lines = 'skip')
    dig_reordered = dig[["Title ISBN"]].merge(dig_api, left_on = "Title ISBN", right_on = "ISBN", how = "left")
    dig_reordered = dig_reordered[~dig_reordered["ISBN"].isna()]
    return dig_reordered

def data_for_classifier(save = True, n = 10000):
    dig = reorder_content_file(qsfile = "data/extract20241127/Extract 20241127_truncated_dig50k.csv",
                                api_output = "data/Output - Digital.csv").head()
    if save:
        dig.to_csv("data/dig_for_class.csv", sep = "|")
    phy = reorder_content_file(qsfile = "data/extract20241127/Extract 20241127_truncated_phy50k.csv",
                                api_output = "data/Output - Physical.csv").head()
    if save:
        phy.to_csv("data/phy_for_class.csv", sep = "|")
    return dig, phy
# combine the LLM outputs

def get_dataframe(fixed_name = "", filetype = "csv", dtype = None):
    df = pd.DataFrame()
    if filetype == "csv":
        for f in glob.glob(fixed_name):
            print(f"loading {f}")
            df = pd.concat([df, pd.read_csv(f, dtype = dtype)])
    if filetype == "excel":
        for f in glob.glob(fixed_name):
            print(f"loading {f}")
            df = pd.concat([df, pd.read_excel(f, dtype = dtype)])
    return df

def get_second_set(dig_llm, dig_reordered, n = 10000):
    dig_no_llm = dig_llm[dig_llm[LLM_TOPIC].isna()| dig_llm[LLM_SUBTOPIC].isna() | dig_llm[LLM_LVL].isna()]
    dig_isbn = dig_no_llm[~dig_no_llm["ISBN"].isna()]
    nlines = dig_no_llm.shape[0] - dig_isbn.shape[0] # how many lines previously sent to LLM but have no isbn
    # need to be replaced by new lines.
    dig_extra = dig_reordered.iloc[(n-nlines):n]
    check_dup = sum(dig_extra["Title ISBN"].isin(dig_llm["Title ISBN"]))
    if check_dup != 0:
        dig_extra = dig_extra[~dig_extra["Title ISBN"].isin(dig_llm["Title ISBN"])]
        dig_extra = pd.concat([dig_extra, dig_reordered.iloc[n:n+check_dup]])
        dig_extra.reset_index(inplace = True, drop = True)
    print(sum(dig_extra["Title ISBN"].isin(dig_llm["Title ISBN"])))
    dig_isbn = dig_isbn.loc[:,dig_extra.columns]
    dig_re = pd.concat([dig_isbn, dig_extra])
    return dig_re


def post_process_keep_valid(save = True):
    dig_llm = get_dataframe(f"data/zz_archive/DIG_processed_interval_*.csv", dtype = object)
    phy_llm = get_dataframe(f"data/zz_archive/PHY_processed_interval_*.csv", dtype = object)
#    dig_llm = dig_llm.loc[~(dig_llm["ISBN"].isna()|dig_llm[LLM_TOPIC].isna()| dig_llm[LLM_SUBTOPIC].isna() | dig_llm[LLM_LVL].isna())]
#    phy_llm = phy_llm.loc[~(phy_llm["ISBN"].isna()|phy_llm[LLM_TOPIC].isna()| phy_llm[LLM_SUBTOPIC].isna() | phy_llm[LLM_LVL].isna())]
    if save:
        dig_llm.to_csv("data/LLM_Output_Digital.csv", index = False)
        phy_llm.to_csv("data/LLM_Output_Physical.csv", index = False)
    return dig_llm, phy_llm

def post_process_extract_extra(dig_llm, phy_llm):
    dig_input, phy_input = data_for_classifier(save = False, n = 10000)
    dig_re = get_second_set(dig_llm, dig_input)
    phy_re = get_second_set(phy_llm, phy_input)
    return dig_re, phy_re

if __name__ == "main":
    # prep data for classifier - output to data/phy_for_class_10k.csv
    dig_input, phy_input = data_for_classifier(save = False, n = 10000)
    
    # after running classifier, use below to check if need to fill gaps where LLM did not return anything
    dig_llm, phy_llm = post_process_keep_valid()

    dig_re, phy_re = post_process_extract_extra(dig_llm, phy_llm)
    dig_re.to_csv("data/dig_for_class_2.csv", sep = "|", index = False)
    phy_re.to_csv("data/phy_for_class_2.csv", sep = "|", index = False)
    
    # after subsequent runs
    dig_llm_final, phy_llm_final = post_process_keep_valid(save = True)