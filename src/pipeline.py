# -*- coding: utf-8 -*-
"""
Full pipeline
"""
import sys
from types import NoneType
from typing import Union, List
import spacy
import time
from tqdm import tqdm
from loguru import logger
from src.preprocess import PreProcessor
from src.entity import EntityExtractor
from src.preprocess import PreProcessor
from src.relation import RelationExtractor
from src.settings import API_KEY_GPT
from src.summary import TextSummarizer
from src.importance_ranking import ImportanceRanker

class CMPipeline:
    """ class for the whole pipeline """

    def __init__(self, options_rel: List[str],
                 preprocess: bool = False,
                 spacy_model: Union[str, None] = None,
                 options_ent: Union[List[str], None] = None,
                 confidence: Union[float, None] = None,
                 db_spotlight_api: Union[str, None] = 'https://api.dbpedia-spotlight.org/en/annotate',
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None,
                 local_rm: Union[bool, None] = None,
                 summary_how: Union[str, None] = None,
                 summary_method: Union[str, None] = None,
                 api_key_gpt: Union[str, None] = None,
                 engine: Union[str, None] = None,
                 temperature: Union[str, None] = None,
                 summary_percentage: Union[str, None] = None,
                 ranking: Union[str, None] = None,
                 ranking_how: Union[str, None] = None,
                 ranking_int_threshold: Union[int, None] = None,
                 ranking_perc_threshold: Union[float, None] = None,
                 word2vec_model_path: Union[str, None] = None):

        # Summary options: 
        # - `single`: summarising each text one by one
        # - `all`: summarising all texts
        self.summary_p = ["single", "all"]
        self.ranking_p = ["single", "all"]
        self.check_params(
            preprocess=preprocess, spacy_model=spacy_model,
            summary_method=summary_method, summary_how=summary_how,
            ranking=ranking, ranking_how=ranking_how
        )

        self.params = {
            "preprocess": {"preprocess": preprocess, "spacy_model": spacy_model,},
            "entity": {"options_ent": options_ent, "confidence": confidence, "db_spotlight_api": db_spotlight_api},
            "relation": {
                "rebel_tokenizer": rebel_tokenizer, "rebel_model": rebel_model, "options_rel": options_rel,
                "local_rm": local_rm},
            "summary": {"summary_method": summary_method, "engine": engine, "temperature": temperature, "summary_percentage": summary_percentage},
            "ranking": {"ranking": ranking, "ranking_how": ranking_how,
                        "ranking_int_threshold": ranking_int_threshold,
                        "ranking_perc_threshold": ranking_perc_threshold}
        }

        self.preprocess = PreProcessor(model=spacy_model) if preprocess else None
        self.entity = EntityExtractor(options=options_ent, confidence=confidence, db_spotlight_api=db_spotlight_api) \
            if options_ent else None
        self.relation = RelationExtractor(
            options=options_rel, rebel_tokenizer=rebel_tokenizer,
            rebel_model=rebel_model, local_rm=local_rm, spacy_model=spacy_model)
        self.nlp = spacy.load(spacy_model)
        self.summary_how = summary_how
        self.summarizer = TextSummarizer(
            method=summary_method, api_key_gpt=api_key_gpt, engine=engine, temperature=temperature, summary_percentage=summary_percentage
        ) if summary_method else None
        self.ranking_how = ranking_how
        self.importance_ranker = ImportanceRanker(ranking=ranking, int_threshold=ranking_int_threshold, perc_threshold=ranking_perc_threshold, word2vec_model_path=word2vec_model_path) \
            if ranking else None

    def check_params(self, preprocess, spacy_model, summary_method, summary_how,
                     ranking, ranking_how):
        """ Check consistency of params """
        if preprocess and (not spacy_model):
            raise ValueError("For preprocessing, you need to enter `spacy_model`")
        if summary_how and summary_how not in self.summary_p:
            raise ValueError(f"For summarisation, `summary_how` should be in {self.summary_p}")
        if ranking and ranking_how not in self.ranking_p:
            raise ValueError(f"For ranking, `ranking_how` should be in {self.ranking_p}")
        if summary_how == 'all' and ranking_how == 'single':
            raise ValueError(f"If `summary_how` is `all`, then `ranking_how` can only be `all`") 

    @staticmethod
    def log_info(message, verbose):
        if verbose:
            logger.info(message)
    
    @staticmethod
    def check_input_summary(input_, summary):
        if summary:
            if isinstance(input_, str) and len(summary) != 1:
                raise ValueError("`input_content` is a string, hence `summary` should have length 1")
            if isinstance(input_, list) and len(input_) != len(summary):
                raise ValueError("`summary` should be the same length as `input_content`")
        return
    
    def __call__(self, input_content: Union[str, List[str]],
                 summaries_list: Union[List[str], None] = None,
                 verbose: bool = False):
        """
        Parameters:
        - input_content: either a string text, or a list of texts to process with the pipeline.
            If string text > converted into list of one text to make it consistent
        - summaries (optional, default to None):
            * must be the same length as input_content
            * summaries[i] is the text input_content[i] that was summarised, and cached beforehand
            * if non null, SKIPPING sentence formatting, preprocessing and summary
        - verbose: whether to output info during the process or no (better for debugging/tracking)
        """
        self.check_input_summary(input_=input_content, summary=summaries_list)

        if isinstance(input_content, str):
            input_content = [input_content]
        start_time = time.time()
        if isinstance(summaries_list, NoneType):  # no cached summaries > doing this on the spot
            # SENTENCE FORMATTING
            docs = [self.nlp(text) for text in input_content]
            docs = [[sent.text.strip() for sent in doc.sents if sent.text.strip()] for doc in docs]

            # PREPROCESSING
            self.log_info(message="Preprocessing", verbose=verbose)
            sentences = [[self.preprocess(x) for x in sentences] for sentences in docs] \
                if self.preprocess else docs
            preprocessing_time = time.time() - start_time
            self.log_info(message="Preprocessing done", verbose=verbose)

            # SUMMARY -> input list of list, outputs sentences_input list of lists
            self.log_info(message="Summary generation", verbose=verbose)
            if self.summarizer:
                summary_generation_start_time = time.time()
                if self.summary_how == "single":  # summarising each document one by one
                    texts = ["\n".join(elt) for elt in sentences]
                    summary = []
                    for text in tqdm(texts):
                        summary.append(self.summarizer(text=text))
                    sentences_input = [self.nlp(text) for text in summary]
                    sentences_input = [[sent.text.strip() for sent in doc.sents if sent.text.strip()] for doc in sentences_input]
                else:  # self.summary_how == "all" -> summarising all documents in one go
                    summary = self.summarizer("\n".join(["\n".join(elt) for elt in sentences]))
                    sentences_input = [self.nlp(summary)]
                    sentences_input = [[sent.text.strip() for sent in doc.sents if sent.text.strip()] for doc in sentences_input]

                summary_generation_time = time.time() - summary_generation_start_time
                logger.info(f"Summary found is :{summary}")
            else:
                summary_generation_time = 0
                sentences_input = sentences
                summary = None
        
        else:  # cached summaries > formatting them to be used as inputs for the rest of the pipeline
            summary = None
            sentences = []
            preprocessing_time = 0
            summary_generation_time = 0
            sentences_input = [self.nlp(text) for text in summaries_list]
            sentences_input = [[sent.text.strip() for sent in doc.sents if sent.text.strip()] for doc in sentences_input]

        # IMPORTANCE RANKING, input sentences_input list of list, output ranked_sents list of str
        sentences_input = [x for x in sentences_input if x]
        # logger.info("SENTENCES:\n"+str(sentences_input[:2]))
        self.log_info(message="Importance Ranking", verbose=verbose)
        if self.importance_ranker:
            ranking_generation_start_time = time.time()
            if self.ranking_how == "single":
                ranked_sents = []
                for sent in tqdm(sentences_input):
                    # logger.info(f"SENTENCES RANKING: {sent}")
                    ranked_sents += self.importance_ranker(sentences=sent)
            if self.ranking_how == "all":
                ranked_sents = self.importance_ranker(sentences=[sent for x in sentences_input for sent in x])
            ranking_extraction_time = time.time() - ranking_generation_start_time
            # logger.info(f"Ranked : {ranked_sents}")

        else:
            ranked_sents = [sent for x in sentences_input for sent in x]
            ranking_extraction_time = 0
        
        # ENTITY EXTRACTION, input entity_input list of str, outputs entities list of str ()
        entity_input = ranked_sents
        self.log_info(message="Entity extraction", verbose=verbose)
        if self.entity:
            entities_start_time = time.time()
            entities = self.entity(text="\n".join(entity_input))
            if "dbpedia_spotlight" in self.params["entity"]["options_ent"]:
                entities["dbpedia_spotlight"] = [x[1] for x in entities["dbpedia_spotlight"]]
            entities = list(set(x for _, v in entities.items() for x in v))

            # logger.info(f"Entities extracted : {entities}")
            entities_extraction_time = time.time() - entities_start_time

        else:
            entities = None
            entities_extraction_time = 0

        # RELATION EXTRACTION
        self.log_info(message="Relation extraction", verbose=verbose)
        # total_time = time.time() - start_time
        relation_extraction_start_time = time.time()
        res = self.relation(sentences=ranked_sents, entities=entities)
        relation_extraction_time = time.time() - relation_extraction_start_time

        total_time = time.time() - start_time

        logger.info(f"Total execution time: {total_time:.4f}s")
        logger.info(f"Preprocessing time: {preprocessing_time:.4f}s")
        logger.info(f"Summary generation time: {summary_generation_time:.4f}s")
        logger.info(f"Ranking extraction time: {ranking_extraction_time:.4f}s")
        logger.info(f"Entity extraction time: {entities_extraction_time:.4f}s")
        logger.info(f"Relation extraction time: {relation_extraction_time:.4f}s")

        text_to_save = "\n".join(["\n".join(x) for x in sentences])
        return [x for _, val in res.items() for x in val], \
            {"text": "\n".join(["\n".join(x) for x in sentences]),
            "entities": entities,"summary": summary,
            "ranked": "\n".join(["\n".join(x) for x in ranked_sents])}


if __name__ == '__main__':
    PIPELINE = CMPipeline(
        preprocess=True, spacy_model="en_core_web_lg",
        options_ent=["dbpedia_spotlight"],
        confidence=0.35,
        db_spotlight_api="http://localhost:2222/rest/annotate",
        options_rel=["rebel"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="./src/fine_tune_rebel/finetuned_rebel.pth", local_rm=True,
        summary_how="single", summary_method="chat-gpt",
        api_key_gpt=API_KEY_GPT, engine="gpt-3.5-turbo",
        summary_percentage=80, temperature=0.0,
        ranking="word2vec", ranking_how="single",
        ranking_perc_threshold=0.8, ranking_int_threshold=None)
    print(PIPELINE.params)
    TEXT = """
    The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
    7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer Hale.
    """
    RES = PIPELINE(input_content=TEXT, verbose=True)
    print(RES[0])
    print("Ranker:")
    print(RES[1]["ranked"])
