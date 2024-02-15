# -*- coding: utf-8 -*-
"""
Relation extractor
"""
import spacy
from typing import Union, List
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.fine_tune_rebel.run_rebel import extract_triples
from src.settings import *


class RelationExtractor:
    """ Extracting relations from text """

    def __init__(self, spacy_model: str, options: List[str] = ["rebel", "dependency"],
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None, local_rm: Union[bool, None] = None):
        """ local_m: whether the model is locally stored or not """
        self.options_p = ["rebel", "dependency"]
        self.options_to_f = {
            "rebel": self.get_rebel_rel,
            "dependency": self.get_dependencymodel

        }
        self.check_params(options=options, rebel_t=rebel_tokenizer,
                          rebel_m=rebel_model, local_rm=local_rm)
        self.params = {
            "options": options,
            "rebel": {
                "tokenizer": rebel_tokenizer,
                "model": rebel_model,
                "local": local_rm
            }
        }
        self.options = options

        if "rebel" in options:
            self.rebel = {
                "tokenizer": AutoTokenizer.from_pretrained(rebel_tokenizer),
                "model": self.get_rmodel(model=rebel_model, local_rm=local_rm),
                "gen_kwargs": {"max_length": 256, "length_penalty": 0,
                               "num_beams": 3, "num_return_sequences": 3, }
            }
        else:
            self.rebel = None

        self.nlp = spacy.load(spacy_model)

    @staticmethod
    def get_rmodel(model: str, local_rm: bool):
        """ Load rebel (fine-tuned or not) model """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        if not local_rm:  # Downloading from huggingface
            model = AutoModelForSeq2SeqLM.from_pretrained(model)
        else:
            model = torch.load(model)
        model.to(device)
        return model

    @staticmethod
    def get_dependencymodel(sentences: str, entities: Union[List[str], None]):
        triplets = []
        for sentence in sentences:
            doc = nlp(sentence)
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass", "agent", "csubjpass",
                                  "csubj", "compound"] and token.head.pos_ in ["VERB", "AUX", "ROOT", "VB", "VBD", "VBG", "VBN", "VBZ"]:
                    subject = token.text
                    verb = token.head.text
                    subject_pos = token.pos_
                    # print(subject_pos)
                    obj = None
                    if any(entity in subject for entity in entities):
                        for child in token.head.children:
                            if child.dep_ in ["dobj", "pobj", "acomp", "attr", "agent", "ccomp", "pcomp",
                                              "xcomp", "csubjpass", "dative", "nmod", "oprd", "obj", "obl"] :
                                obj = child.text
                                obj_pos = child.pos_
                                # print(obj_pos)

                                if subject_pos in ["NOUN", "PROPN"] and obj_pos in ["NOUN", "PROPN"]:
                                    triplets.append((subject, verb, obj))
                    else:
                        for child in token.head.children:
                            if child.dep_ in ["dobj", "pobj", "acomp", "attr", "agent", "ccomp", "pcomp",
                                              "xcomp", "csubjpass", "dative", "nmod", "oprd", "obj", "obl"] :
                                obj = child.text
                                obj_pos = child.pos_
                                # print(obj_pos)
                                if subject_pos in ["NOUN", "PROPN","ADP"] and obj_pos in ["NOUN", "PROPN"] and any(entity in obj for entity in entities):
                                    triplets.append((subject, verb, obj))
        return triplets

    def check_params(self, options, rebel_t, rebel_m, local_rm):
        """ Check that each parameter is correct for the options """
        if any(x not in self.options_p for x in options):
            raise ValueError(f"All options in `options` must be from {self.options_p}")

        if "rebel" in options:
            if any(not isinstance(x, y) for (x, y) in \
                   [(rebel_t, str), (rebel_m, str), (local_rm, bool)]):
                raise ValueError("To extract relations with REBEL, you need to specify: " + \
                                 "`rebel_tokenizer` as string, `rebel_model` as string, `local_rm` as bool")

    def predict(self, input_m):
        """ Text > predict > human-readable """
        for key in ["input_ids", "attention_mask"]:
            if len(input_m[key].shape) == 1:
                #  Reshaping, has a single sample
                input_m[key] = input_m[key].reshape(1, -1)

        output = self.rebel['model'].generate(
            input_m["input_ids"].to(self.rebel['model'].device),
            attention_mask=input_m["attention_mask"].to(self.rebel['model'].device),
            **self.rebel['gen_kwargs'], )

        decoded_preds = self.rebel['tokenizer'].batch_decode(output, skip_special_tokens=False)
        return decoded_preds

    def get_dataloader(self, sent_l: List[str], batch_size: int = 16):
        if not sent_l:
            return None
        sent_l = [x for x in sent_l if x]
        sent_l = [x for x in sent_l if len(x.split()) <= 256]
        
        dataset = Dataset.from_dict({"text": sent_l})
        dataset = dataset.map(lambda examples: self.rebel['tokenizer'](examples["text"], max_length=256, padding=True, truncation=True, return_tensors='pt'), batched=True)
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])
        return DataLoader(dataset, batch_size=batch_size)

    def get_rebel_rel(self, sentences: List[str], entities: Union[List[str], None]):
        """ Extracting relations with rebel """

        # input_m = self.tokenize(text=sentences)
        dataloader = self.get_dataloader(sent_l=sentences)
        if not dataloader:  # empty sentences
            return []
        output_m = []
        for batch in dataloader:
            try:
                output_m += self.predict(input_m=batch)
            except:
                pass

        unique_triples_set = set()  # Set to store unique triples
        res = []

        if not entities:
            for x in output_m:
                for triple in self.post_process_rebel(x):
                    if triple not in unique_triples_set:
                        res.append(triple)
                        unique_triples_set.add(triple)
        else:
            for entity in entities:
                entity_strings = [item for tuple_item in entities for item in tuple_item]
                cands = [x for x in output_m if any(entity_string in x for entity_string in entity_strings)]

                for x in cands:
                    for triple in self.post_process_rebel(x):
                        if triple not in unique_triples_set:
                            res.append(triple)
                            unique_triples_set.add(triple)

        return res

    @staticmethod
    def post_process_rebel(x):
        """ Clean rebel output"""
        res = extract_triples(x)
        return [(elt['head'], elt['type'], elt['tail']) for elt in res]

    def __call__(self, sentences: List[str], entities: Union[List[str], None] = None):
        """ Extract relations for one string text """
        res = {}
        for option in self.options:
            curr_res = self.options_to_f[option](sentences=sentences, entities=entities)
            curr_res = [x for x in curr_res if x[0].lower() != x[2].lower()]
            res[option] = list(set(curr_res))
        return res


if __name__ == '__main__':
    REL_EXTRACTOR = RelationExtractor(
        options=["rebel"], rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="./src/fine_tune_rebel/finetuned_rebel.pth", local_rm=True,
        spacy_model="en_core_web_lg")
    from nltk.tokenize import sent_tokenize
    from entity import *

    ENTITY_EXTRACTOR = EntityExtractor(options=["dbpedia_spotlight"], confidence=0.35,
                                       db_spotlight_api="http://localhost:2222/rest/annotate")
    folder_path = WIKI_TRAIN + "/116"
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                RES = ENTITY_EXTRACTOR(text=text)
                extracted_strings_2 = [item[1] for item in RES['dbpedia_spotlight']]
                print("## ENTITIES")
                print(extracted_strings_2)
                sentences = sent_tokenize(text)
                print("## SENTENCES")
                print(sentences)
                RES = REL_EXTRACTOR(sentences=sentences, entities=extracted_strings_2)
                print("## RELATION")
                print(RES)
