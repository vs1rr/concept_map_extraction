from typing import Union, List
import requests
from src.settings import *

class EntityExtractor:
    """ Extracting entities from text """

    def __init__(self, options: List[str] = ["dbpedia_spotlight"],
                 confidence: Union[float, None] = None,
                 db_spotlight_api: str = 'https://api.dbpedia-spotlight.org/en/annotate'):
        """ Init main params
        - options: how to extract entities

        Default: calls Spotlight API
        Custom: using local spacy model """
        self.options_p = ["dbpedia_spotlight", "spacy"]
        self.options_to_f = {
            "dbpedia_spotlight": self.get_dbs_ent,
            "spacy": self.get_spacy_ent,

        }
        self.check_params(options=options, confidence=confidence)

        self.params = {
            "options": options,
            "confidence": confidence,
            "db_spotlight_api": db_spotlight_api
        }
        self.options = options

        # DBpedia Spotlight params
        self.confidence = confidence
        self.headers = {'Accept': 'application/json'}
        self.dbpedia_spotlight_api = db_spotlight_api
        self.timeout = 3600

    def check_params(self, options, confidence):
        """ Check that each parameter is correct for the options """
        if any(x not in self.options_p for x in options):
            raise ValueError(f"All options in `options` must be from {self.options_p}")

        if "dbpedia_spotlight" in options:
            if not isinstance(confidence, float):
                raise ValueError("To extract entities with DBpedia Spotlight, " + \
                                 "you need to specify `confidence` as a float")

    def get_dbs_ent(self, text: str):
        """ Retrieve entities with Spotlight """
        response = requests.post(
            self.dbpedia_spotlight_api, data=self.get_payload(text=text),
            headers=self.headers, timeout=self.timeout)
        if response.status_code == 200:
            try:
                return set([(resource["@URI"], resource["@surfaceForm"]) \
                            for resource in response.json()["Resources"]])
            except:
                return set()
        return set()

    def get_spacy_ent(self, text: str):
        doc = nlp(text)
        found_spacy_entities_set = set()

        for ent in doc.ents:
            found_spacy_entities_set.add(ent.text.lower())
        found_spacy_entities_set = list(found_spacy_entities_set)
        return found_spacy_entities_set

    def get_payload(self, text: str):
        """ Payload for requests """
        return {'text': text, 'confidence': self.confidence}

    def __call__(self, text: str):
        """ Extract entities for one string text """
        res = {}
        for option in self.options:
            entities = self.options_to_f[option](text=text)
            res[option] = entities

        return res


if __name__ == '__main__':
    ENTITY_EXTRACTOR = EntityExtractor(options=["dbpedia_spotlight"], confidence=0.35,
                                       db_spotlight_api="http://localhost:2222/rest/annotate")
    TEXT = """
    The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
    7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer
    Hale.
    """
    RES = ENTITY_EXTRACTOR(text=TEXT)
    print(RES)
