# -*- coding: utf-8 -*-
"""
Get aggregated results from all experiments
"""
import os
import json
import scipy
import numpy as np
import pandas as pd
from loguru import logger
from src.build_table import build_table
####### PARAMS BELOW TO UPDATE
SAVE_FOLDER = "./experiments"
DATA_PATH = "./src/data/Corpora_Falke/Wiki/train/"
FOLDERS_CMAP = [x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))]
DATE_START = "2024-02-08-11:00:00"
####################

COLUMNS = [
    'summary_method', 'summary_percentage',
    'ranking', 'ranking_how', 'ranking_perc_threshold',
    'confidence',
    'relation',
    'meteor_pr', 'meteor_re', 'meteor_f1',
    'rouge-2_pr', 'rouge-2_re', 'rouge-2_f1'
]

def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as openfile:
        data = json.load(openfile)
    return data

def avg_results(metrics):

    res = {
        x+"_"+y: [] for x in ["meteor", "rouge-2"] for y in ["pr", "re", "f1"]
    }

    for _, info in metrics.items():
        for k1, val in info.items():
            for k2, metric in val.items():
                res[f"{k1}_{k2[:2]}"].append(metric)
    
    for k1, v in res.items():
        res[k1] = round(np.mean(v), 1)

    return res

def get_folders_exp_finished():
    exps = os.listdir(SAVE_FOLDER)
    exps = [x for x in exps if \
        all(y in os.listdir(os.path.join(SAVE_FOLDER, x)) \
            for y in ["metrics.json", "params.json", "logs.json"] + FOLDERS_CMAP) and \
                x >= DATE_START]
    exps = [
        (read_json(os.path.join(SAVE_FOLDER, x, "logs.json")),
         read_json(os.path.join(SAVE_FOLDER, x, "params.json")),
         read_json(os.path.join(SAVE_FOLDER, x, "metrics.json"))) for x in exps]
    exps = [x for x in exps if x[0].get("finished") == "yes"]
    return [x[1:] for x in exps]

def get_rebel_opt(params):
    options_rel = params["relation"]["options_rel"]
    local_rm = params["relation"]["local_rm"]
    if "rebel" in options_rel:
        x1 = "rebel\\_ft" if local_rm else "rebel\\_hf"
        x2 = "+dependency" if "dependency" in options_rel else ""
        return x1+x2
    return "+".join(options_rel)

def format_vals(input, val1):
    return [1 if x == val1 else 0 for x in input]

def get_correlations(df_, feat_cols, metric_cols):
    mappings = {
        "summary_method": {1: "chat-gpt", 2: "lex-rank"},
        "ranking_how": {1: "all", 2: "single"},
        "confidence": {1: 0.5, 2: 0.7},
        "relation": {1: "rebel\_ft", 2: "rebel\_hf"},
        
    }

    cols_df_corr_binary = ["Feature", "Val1", "Val2", "Metric", "Correlation", "Pvalue"]
    df_corr_binary = pd.DataFrame(columns=cols_df_corr_binary)
    for x, info in mappings.items():
        for metric in ["meteor_f1", "rouge-2_f1"]:
            # print(f"FEATURE: {x} | METRIC {metric}")
            vals_1 = format_vals(df_[x], info[1])
            vals_2 = df_[metric]
            corr, pvalue = scipy.stats.spearmanr(vals_1, vals_2)
            # print(f"Corr: {corr} | Pvalue: {pvalue}")
            curr_l = [x.replace("_", "\\_"), 
                      info[1], info[2], metric.replace("_", "\\_"),
                      round(corr, 3), "{:.2e}".format(pvalue)]
            df_corr_binary.loc[len(df_corr_binary)] = curr_l
    print("===========\nCORRELATIONS BINARY VALUES")
    latex_table = build_table(
        columns=["Feature", "Metric", "Values", "Correlation", "Pvalue"],
        alignment="r"*len(cols_df_corr_binary),
        caption="Correlation between comparison of features and F1 scores",
        label="tab:wiki-train-binary-feat-corr-pval",
        position="h",
        data=df_corr_binary.values,
        sub_columns=["", "", "Val1", "Val2", "", ""],
        multicol=[1, 1, 2, 1, 1],
        # sub_columns=[x.replace("_", "\\_") for x in COLUMNS[:6]] + ["Pr", "Re", "F1"]*2,
        # multicol=[2, 2, 1, 1, 3, 3],
        resize_col=2
    )
    print(f"{latex_table}\n=====")

    rankings = [
        ["page\_rank", "word2vec"],
        ["page\_rank", "tfidf"],
        ["word2vec", "tfidf"]]
    for [r_a, r_b] in rankings:
        for metric in ["meteor_f1", "rouge-2_f1"]:
            print(f"FEATURE: ranking | METRIC {metric} | VAL1 {r_a} | VAL2 {r_b}")
            curr_df = df_[df_.ranking.isin([r_a, r_b])]
            vals_1 = format_vals(curr_df.ranking, r_a)
            vals_2 = curr_df[metric]
            corr, pvalue = scipy.stats.spearmanr(vals_1, vals_2)
            print(f"Corr: {corr} | Pvalue: {pvalue}")
    print("==========")
    for r_a in ["page\_rank", "word2vec", "tfidf"]:
        for metric in ["meteor_f1", "rouge-2_f1"]:
            print(f"FEATURE: ranking | METRIC {metric} | VAL1 {r_a} | VAL2 Other")
            vals_1 = format_vals(df_.ranking, r_a)
            vals_2 = df_[metric]
            corr, pvalue = scipy.stats.spearmanr(vals_1, vals_2)
            print(f"Corr: {corr} | Pvalue: {pvalue}")

def f1_helper(row):
    if np.isnan(row["meteor_f1"]):
        return 2*row["meteor_re"]*row["meteor_pr"]/(row["meteor_re"]+row["meteor_pr"])
    return row["meteor_f1"]

def main():
    df_output = pd.DataFrame(columns=COLUMNS)
    folders_exp = get_folders_exp_finished()
    logger.info(f"Results on {len(folders_exp)} experiments")

    for params, metrics in folders_exp:
        avg_metrics = avg_results(metrics)
        curr_l = [
            params["summary"]["summary_method"].replace("_", "\\_"),
            params["summary"]["summary_percentage"],
            params["ranking"]["ranking"].replace("_", "\\_"),
            params["ranking"]["ranking_how"],
            params["ranking"]["ranking_perc_threshold"] * 100,
            params["entity"]["confidence"],
            get_rebel_opt(params)
        ] + \
            [avg_metrics[f"{x}_{y}"] for x in ["meteor", "rouge-2"] \
                for y in ["pr", "re", "f1"]]
        df_output.loc[len(df_output)] = curr_l
        # df_output = df_output.append(pd.Series(curr_l, index=COLUMNS), ignore_index=True)
    print(df_output)
    df_output.meteor_f1 = df_output.apply(f1_helper, axis=1)
    df_output.sort_values(by=COLUMNS[:6]).to_csv("experiments_acl/results.csv")

    # latex_table = build_table(
    #     columns=["Summary", "Ranking", "Entity", "Relation", "METEOR", "ROUGE-2"],
    #     alignment="r"*len(COLUMNS),
    #     caption="Results for all systems on Wiki TRAIN",
    #     label="res-wiki-train-all-hyperparams",
    #     position="h",
    #     data=df_output.sort_values(by=COLUMNS[:7]).values,
    #     sub_columns=[x.replace("_", "\\_") for x in COLUMNS[:7]] + ["Pr", "Re", "F1"]*2,
    #     multicol=[2, 3, 1, 1, 3, 3],
    #     resize_col=2
    # )
    # print(latex_table)

    get_correlations(df_=df_output, feat_cols=COLUMNS[:6], metric_cols=["meteor_f1", "rouge-2_f1"])
    
    



if __name__ == '__main__':
    main()