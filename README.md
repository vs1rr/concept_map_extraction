## Unlocking Understanding: A Novel Pipeline for Automated Concept Map Extraction from Text

### <center> Anonymised Code for Paper submitted to ACL 2024 - Call for Main Conference Papers </center> ###

## Quick Start
### Set Up Virtual Environment

(All scripts are run from the root directory of this repository)

We run our code on both Mac M1 and Ubuntu machines, with conda for virtual environment and python version 3.11.3.

To install the requirements
```bash
pip install -r requirements.txt
```

If one of the dependency yield error, you can alternatively run the following Python script which will install them one by one:
```bash
python install_req.py
```

To run the pipeline, you need to have an OpenAI account with the API token. In the `src/settings.py` file, update it in the `API_KEY_GPT` variable.

You can then set up the package
```bash
python setup.py install
```

Lastly, you need to pre-download some resources:
* From Terminal
```bash
pip install coreferee python -m coreferee install en
```

* From Python
```python
import nltk
nltk.download('punkt')
```

### Running the Pipeline

The main file to run the pipeline is `src/pipeline.py`. More details on each component can be found below.

#### Code structure

Main components
* `data_load.py`. Main class `DataLoader` to handle data load for one folder, parameters:
    * `path`: path folder to the data
    * `type_d`: either __single__ or __multi__
    * `one_cm`: boolean, whether `path` points, to one concept map or several ones
    * `summary_path`: (optional) path to the pre-stored summaries
* `preprocess.py`. Main class `PreProcessor`. parameters:
    * `model`: spaCy model to use
* `summary.py`. Main class `TextSummarizer`, parameters:
    * `method`: either __lex-rank__ or __chat-gpt__
    * `api_key_gpt`: (only if `method == "chat-gpt"`) API key to OpenAI
    * `engine`: (only if `method == "chat-gpt"`) model to use
    * `temperature`: temperature for the model
    * `summary_percentage`: `int` between 0 and 100
* `importance_ranking.py`. Main class `ImportanceRanker`, parameters:
    * `ranking`: either __page_rank__, __tfidf__ or __word2vec__
    * `int_threshold` and `perc_threshold`: either integer for the number of sentences to keep, or float between 0 and 1
* `entity.py`. Main class `EntityExtractor`, parameters:
    * `options`: sublist, list of elements in the following list: `["dbpedia_spotlight", "spacy"]`
    * `confidence`: confidence level for entity extraction
    * `db_spotlight_api`: DBpedia Spotlight API for entity extraction.
        * The default one is the one publicly available: `https://api.dbpedia-spotlight.org/en/annotate`
        * For more efficiency we strongly recommend to set up your local API. We followed instructions on [this link](https://github.com/MartinoMensio/spacy-dbpedia-spotlight).
* `relation.py`. Main class `RelationExtractor`, parameters:
    * `spacy_model`: spaCy model to use
    * `options`: sublist, list of elements in the following list: `["rebel", "dependency"]`
    * `rebel_tokenizer`: (if `rebel` in `options`) hugginface tokenizer to use
    * `rebel_model`: rebel model to use, either from hugginface, or local torch model
    * `local_rm`: boolean, whether the model is local (torch model) or not (huggingface model)
* `pipeline.py`. Main class `CMPipeline`, parameters:
    * For preprocessing: `preprocessing`, `spacy_model`
    * For summary: `summary_method`, `api_key_gpt`, `engine`, `temperature`, `summary_percentage`
    * For importance ranking: `ranking`, `ranking_int_threshold`, `ranking_perc_threshold`
    * For entity: `options_ent`, `confidence`, `db_spotlight_api`
    * For relation: `options_rel`, `rebel_tokenizer`, `rebel_model`, `local_rm`
    * For pipeline:
        * `summary_how`: either __single__ (document-level) or __all__ (all document level)
        * `ranking_how`: either __single__ (document-level) or __all__ (all document level)
* `evaluation.py`: main class `EvaluatinMetrics`
* `experiment.py`: main class `ExperimentRun`, same parameters as `pipeline`, and additional ones to initialize the data loader

Miscellaneous
* `build_table.py`: helpers for building overleaf tables
* `settings.py`: main settings (main one used is the OpenAI API token)

### Data used

For our experiments, we reached out to Falke via email, requesting access to the WIKI and BIO corpus, along with its train and test split as originally utilized in the following publications:
* Falke, T. (2019). Automatic structured text summarization with concept maps.
* Falke, T., Meyer, C. M., & Gurevych, I. (2017, November). Concept-map-based multi-document summarization using concept coreference resolution and global importance optimization. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 801-811).

Regrettably, we are unable to provide access to the corpus as it is not within our ownership. Nevertheless, we have made available the summaries generated by ChatGPT 3.5 Turbo and LexRank in the `src/summaries` folder, divided into `test` and `train` subsets as explained in the Reproducibility Section. 

### Reproducibility

* For our experiments, we fine-tuned the REBEL model available on [Hugging Face](https://huggingface.co/Babelscape/rebel-large). All the related code can be found in the `src/fine_tune_rebel` folder, including a more detailed [notes](./src/fine_tune_rebel/notes.md). The fine-tuned model is big and can be sent upon demand.

* We then several experiments on the WIKI datasets. All the scripts used can be found in the `experiments_acl` folder:
    * `make_summaries.py`: retrieve and store summaries from OpenAI
    ```bash
    python experiments_acl/make_summaries.py <data-path> <output-folder> <type-data> <dataset>
    ```
    * `run_train.py`: run all experiments for the hyperparameters. Update parameters on the top of the file, and run:
    ```bash
    python experiments_acl/run_train.py
    ```
    * `get_results.py`: analyse results for the hyperparameter tuning part. Update parameters on the top of the file, and run
    ```bash
    python experiments_acl/get_results.py
    ```
    * `results.csv`: results for hyperparameter tuning
    * `run_single_and_ablation.py`: running final experiments (all the components and ablation studies)
    ```bash
    python experiments_acl/run_single_and_ablation.py <data-path> <save-folder> <summary-folder>
    ```
    * `wiki_test` and `wiki_train`: results on the test and train sets of WIKI
    * `get_final_results.py`: analyse final results
    ```bash
    python experiments_acl/get_final_results.py experiments_acl/wiki_train
    python experiments_acl/get_final_results.py experiments_acl/wiki_test
    ```

* We also make the summaries that we used available in the `summaries` folder.


### License

Distributed under the terms of the Apache License 2.0.
