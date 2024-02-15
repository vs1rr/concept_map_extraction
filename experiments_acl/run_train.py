# -*- coding: utf-8 -*-
"""
Run experiments on the Wiki train dataset
--> testing different hyperparameters
"""
import os
import json
from datetime import datetime
from loguru import logger
from sklearn.model_selection import ParameterGrid
from src.settings import API_KEY_GPT
from src.experiment import ExperimentRun

####### PARAMS BELOW TO UPDATE 
SAVE_FOLDER = "./experiments"
HF_RM_MODEL = "Babelscape/rebel-large"
LOCAL_RM_MODEL = "./src/fine_tune_rebel/finetuned_rebel.pth"
# Date from which to consider the folders in the experiments 
# (to check whether this parameters have been run already or not)
DATE_START = "2024-02-08-11:00:00"
DATA_PATH = "./src/data/Corpora_Falke/Wiki/train/"
FOLDERS_CMAP = [x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))]
TYPE_DATA = "multi"
ONE_CM = False
SUMMARY_FOLDER = "summaries"
####################

FIXED_PARAMS = {
    "preprocess": True,
    "spacy_model": "en_core_web_lg",
    "options_ent": ["dbpedia_spotlight"],
    "db_spotlight_api": "http://localhost:2222/rest/annotate",
    "rebel_tokenizer": "Babelscape/rebel-large",
    "summary_how": "single",
    "api_key_gpt": API_KEY_GPT,
    "engine": "gpt-3.5-turbo",
    "temperature": 0.0,
    # "ranking_how": "single"
}


VARIABLE_PARAMS = {
    # Summarisation
    "summary_percentage": [15, 30, 50, 70],
    "summary_method": ["lex-rank", "chat-gpt"],
    # Importance ranking
    "ranking": ["word2vec", "page_rank", "tfidf"],
    "ranking_how": ["single", "all"],
    "ranking_perc_threshold": [0.15, 0.3, 0.5, 0.7],
    # Entity
    "confidence": [0.5, 0.7],
    # Relation extraction
    "options_rel": [["rebel"]],
    "local_rm": [True, False],
    "rebel_model": [HF_RM_MODEL, LOCAL_RM_MODEL],
}

def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as openfile:
        data = json.load(openfile)
    return data

def get_folders_exp_finished():
    exps = os.listdir(SAVE_FOLDER)
    exps = [x for x in exps if \
        all(y in os.listdir(os.path.join(SAVE_FOLDER, x)) \
            for y in ["metrics.json", "params.json", "logs.json"] + FOLDERS_CMAP) and \
                x[:19] >= DATE_START]
    exps = [(x, read_json(os.path.join(SAVE_FOLDER, x, "params.json")), read_json(os.path.join(SAVE_FOLDER, x, "logs.json"))) for x in exps]
    exps = [x for x in exps if x[2].get("finished") == "yes"]
    return [x[:2] for x in exps]

def format_params(params):
    """ Changed stored params into format comparable to ParameterGrid """
    res = {}
    for k1, v1 in params.items():
        for k2, v2 in {x: y for x, y in v1.items() if x in VARIABLE_PARAMS}.items():
            res[k2] = v2
    return res

def get_params_to_run(filtered_params):
    folders = get_folders_exp_finished()
    run_exps = [format_params(x[1]) for x in folders]
    run_exps = [x for x in run_exps if x]
    return [x for x in filtered_params if x not in run_exps]

def init_exp(params):
    experiment = ExperimentRun(
        # DATA INFO
        folder_path=DATA_PATH, type_data=TYPE_DATA, one_cm=ONE_CM,
        summary_path=os.path.join(
            SUMMARY_FOLDER, params["summary_method"], str(params["summary_percentage"])),
        # PREPROCESS
        preprocess=FIXED_PARAMS["preprocess"],
        spacy_model=FIXED_PARAMS["spacy_model"],
        # SUMMARY
        summary_how=FIXED_PARAMS["summary_how"],
        summary_method=params["summary_method"],
        api_key_gpt=API_KEY_GPT if params["summary_method"] == "chat-gpt" else None,
        engine=FIXED_PARAMS["engine"] if params["summary_method"] == "chat-gpt" else None,
        temperature=FIXED_PARAMS["temperature"] if params["summary_method"] == "chat-gpt" else None,
        summary_percentage=params["summary_percentage"],
        # IMPORTANCE RANKING
        ranking=params["ranking"],
        ranking_how=params["ranking_how"],
        ranking_perc_threshold=params["ranking_perc_threshold"],
        # ENTITY
        options_ent=FIXED_PARAMS["options_ent"],
        confidence=params["confidence"],
        db_spotlight_api=FIXED_PARAMS["db_spotlight_api"],
        # RELATION EXTRACTION
        options_rel=params["options_rel"],
        rebel_tokenizer=FIXED_PARAMS["rebel_tokenizer"] if "rebel" in params["options_rel"] else None,
        rebel_model=params["rebel_model"] if "rebel" in params["options_rel"] else None,
        local_rm=params["local_rm"] if "rebel" in params["options_rel"] else None,
    )
    return experiment

def run_one_exp(params):
    experiment = init_exp(params)
    try:
        experiment(save_folder=SAVE_FOLDER)
    except Exception as e:
        logs_txt = "logs_exp_run.txt"
        f_log = open(logs_txt, "a" if os.path.exists(logs_txt) else "w")
        f_log.write("==========\n" + f"DATE: {str(datetime.now())}" + "\n")
        f_log.write("Could not run experiment for the following params:\n" + str(params) + "\n\nException: \n" + str(e) + "\n==========\n\n")
        f_log.close()


if __name__ == '__main__':
    PARAMS = list(ParameterGrid(VARIABLE_PARAMS))

    FILTERED_PARAMS = [x for x in PARAMS if \
        (("rebel" in x["options_rel"]) and x["local_rm"] and x["rebel_model"] == LOCAL_RM_MODEL) or \
            (("rebel" in x["options_rel"]) and x["local_rm"] == False and x["rebel_model"] == HF_RM_MODEL)]
    PARAMS_TO_RUN = get_params_to_run(filtered_params=FILTERED_PARAMS)
    PERC = round(100*len(PARAMS_TO_RUN)/len(FILTERED_PARAMS))
    logger.info(f"{len(FILTERED_PARAMS)} set of parameters to be run in total, still {len(PARAMS_TO_RUN)}({PERC}%) to go")

    for params in PARAMS_TO_RUN:
        run_one_exp(params=params)