import os
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from spacy.tokens import Doc
import pandas as pd

def get_entities(doc: Doc) -> list[str]:
    """ Getting dbpedia entities from spacy doc """
    res = [ent for ent in doc.ents if ent._.dbpedia_raw_result]
    return list(set(ent._.dbpedia_raw_result["@URI"] for ent in res))

def extract_dbpedia_spotlight_entities(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as file:
                    text = file.read()
                    payload = {
                        'text': text,
                        'confidence': 0.35
                    }
                    headers = {'Accept': 'application/json'}
                    response = requests.post('https://api.dbpedia-spotlight.org/en/annotate', data=payload, headers=headers)
                    if response.status_code == 200:
                        api_result = response.json()
                        entities = [(resource["@URI"], resource["@surfaceForm"]) for resource in api_result["Resources"]]
                        print(entities)  # Print entities found by the API
                        output_root = root.replace(input_folder, output_folder)
                        os.makedirs(output_root, exist_ok=True)
                        output_file_name = file_name.replace(".txt", "-entities.txt")  # Change the extension to .txt
                        output_file_path = os.path.join(output_root, output_file_name)
                        with open(output_file_path, "w") as openfile:
                            for entity, surface_form in entities:
                                openfile.write(f"{surface_form} - {entity}\n")  # Write original word and entity to a new line in the text file
                    else:
                        print(f"Error: {response.status_code} - Failed to get entities for {file_path}")


def extract_dbpedia_spotlight_entities_bio(input_text, output_file_path):
    payload = {
        'text': input_text,
        'confidence': 0.35
    }
    headers = {'Accept': 'application/json'}
    response = requests.post('https://api.dbpedia-spotlight.org/en/annotate', data=payload, headers=headers)

    if response.status_code == 200:
        api_result = response.json()
        entities = [(resource["@URI"], resource["@surfaceForm"]) for resource in api_result["Resources"]]
        print(entities)  # Print entities found by the API

        with open(output_file_path, "w") as openfile:
            for entity, surface_form in entities:
                openfile.write(f"{surface_form} - {entity}\n")  # Write original word and entity to a new line in the text file
    else:
        print(f"Error: {response.status_code} - Failed to get entities")

def process_csv_bio(csv_path, output_folder):
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return

    for index, row in df.iterrows():
        try:
            context = row['context']
            triplets = row['triplets']

            # Assume you want to create separate output files for each row
            output_file_name = f"output_{index + 1}-entities.txt"
            output_file_path = os.path.join(output_folder, output_file_name)

            # Apply the entity extraction function
            extract_dbpedia_spotlight_entities(context, output_file_path)
        except Exception as e:
            print(f"Error processing row {index + 1}: {e}")
