import os
import re
from datetime import date
import pandas as pd
import json
from datetime import datetime
import requests

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from chunking import langchain_split
import time
import concurrent.futures
import argparse
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='Startive')
parser.add_argument('--file_path', type=str, default='./sample-files/SCD-Tej.txt',
                    help='the data path of file to be converted to vector embbedings')
parser.add_argument('--splitter', type=str, default='HTMLTextSplitter',
                    help='splitter tp split document')
parser.add_argument('--input_text', type=str, default=None,
                    help='input_text to split')
parser.add_argument('--ES_CLOUD_ID', type=str, default=None,
                    help='Elastic Search CLOUD_ID')
parser.add_argument('--ES_API_KEY', type=str, default=None,
                    help='Elastic Search API_KEY')
parser.add_argument('--ELSER_Model', type=str, default=True,
                    help='Set True if you want to use Elser Elastic serach model')
parser.add_argument('--index_name', type=str, default='Index1',
                    help='input ES index_name')
parser.add_argument('--index_mapping', type=str, default='./sparse_encoder_1.txt',
                    help='index_mapping file path')
parser.add_argument('--ES_username', type=str, default=None,
                    help='elastic')
parser.add_argument('--ES_password', type=str, default=None,
                    help='11ilk50GgQJZ17RV7Zu7b2R0')


args = parser.parse_args()

splitter_arg = args.splitter
input_text = args.input_text
file_path = args.file_path
try:
    file_name = file_path.split('/')[-1]
except Exception as e:
    print("An error occurred:", e)

# ES_CLOUD_ID="cfe0bcaa729d4702bbd84e5becdb08f7:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGQxZDE0MTZkNzIzMjQ0YWZiNDY0ZGJkNjEwYTAwZTIzJGFmNzNjYThhZTEyZTQ3ODliYjEyZjdlNGQ4ODNlMjAz"
# ES_API_KEY="VWo4cjY0OEIyYUpJZzFTa09DbzM6YUpVOHl2TE5SZW1DaXJKajZtVzdBdw=="
# ES_CLOUD_ID=args.ES_CLOUD_ID #"9ceb45cca61b4ab1abcb2dc0b6132e7f:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGUzNTZiMTk2NDJiMzRiMTg5NmQ4NTIxODAzY2M3NzNlJGU2MzQ0NzhkN2M5YTRmZWY4ZWM5OGYzNmE4NDM4MDVh"
# ES_API_KEY=args.ES_API_KEY #"ZmtZY1VKQUJDdjUzXzZ2bzVMTUE6T0lHUHZnUWRUb3ExYzJpSDk2TTdiUQ=="
es_username = args.ES_username
es_password = args.ES_password

use_ELSER_model = args.ELSER_Model
## Replace elastic instance here
es_client = Elasticsearch(f"https://kb.strative.ai/esuser:{es_username}espassword8:{es_password}@eshost:port",  ca_certs=False, verify_certs=False)

es_client = Elasticsearch(
    cloud_id=ES_CLOUD_ID,
    api_key=ES_API_KEY,
    request_timeout=30,
    max_retries=10,
    retry_on_timeout=True,
)

print (es_client.info())

if use_ELSER_model:
    # es_client.ml.put_trained_model(
    #     model_id=".elser_model_2", input={"field_names": ["text_field"]}
    # )

    # while True:
    #     status = es_client.ml.get_trained_models(
    #         model_id=".elser_model_2", include="definition_status"
    #     )

    #     if status["trained_model_configs"][0]["fully_defined"]:
    #         print("ELSER Model is downloaded and ready to be deployed.")
    #         break
    #     else:
    #         print("ELSER Model is downloaded but not ready to be deployed.")
    #     time.sleep(5)

    # es_client.ml.start_trained_model_deployment(
    #     model_id=".elser_model_2", number_of_allocations=1, wait_for="starting"
    # )

    while True:
        status = es_client.ml.get_trained_models_stats(
            model_id=".elser_model_2",
        )
        if status["trained_model_stats"][0]["deployment_stats"]["state"] == "started":
            print("ELSER Model has been successfully deployed.")
            break
        else:
            print("ELSER Model is currently being deployed.")
        time.sleep(5)

    es_client.ingest.put_pipeline(
        id="nq-ingest-pipeline",
        description="Ingest pipeline for NQ dataset",
        processors=[
            {
                "inference": {
                    "model_id": ".elser_model_2",
                    "input_output": [
                        {"input_field": "document_text", "output_field": "document_text_embedding"}
                    ],
                }
            }
        ],
    )

## Download model for KNN
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

## create the index
def create_index(index_name,mapping):
    try:
        es_client.indices.create(index=index_name,body = mapping)
        print(f"Index '{index_name}' created successfully.")
    except RequestError as e:
        if e.error == 'resource_already_exists_exception':
            print(f"Index '{index_name}' already exists.")
        else:
            print(f"An error occurred while creating index '{index_name}': {e}")

def delete_index(index_name):
    es_client.indices.delete(index=index_name, ignore_unavailable=True)

def index_data(df_docs,source,index_name_elser,index_name_knn):
    docs = df_docs.iloc[:100].iterrows()
    for index, row in docs:
        # i=i+1
        print("Processing i",index)
        example_id = row['example_id']
        document_title = 'document_title'#row['document_title']
        document_url = row['document_url']
        document_text = row['document_text'] # avail in simplified nq

        splitter = "HTMLTextSplitter"
        text_chunks = langchain_split.split_input(splitter, None, document_text)
        # text_chunks = [document_text]

        for i,chunk in enumerate(text_chunks):
            chunk_text_dense_embedding = model.encode(chunk) #replace this with document_text (from simplified nq)
            #When using ELSER document_text_embedding is populated through inference pipeline
            doc_elser ={
                            "document_text": chunk,
                            "document_title" : f'{i}_{document_url}',
                            "example_id": f'{example_id}',
                            "document_url": document_url,
                            "source": source
                }
            
            doc_knn ={
                            "document_text": chunk,
                            "document_title" : document_title,
                            "example_id": example_id,
                            "document_url": document_url,
                            "source": source,
                            "document_text_dense_embedding": chunk_text_dense_embedding
                    }
            response = es_client.index(index=index_name_elser, body=doc_elser)
            # print(response)
            response = es_client.index(index=index_name_knn, body=doc_knn)
            # print(response)           

def index_data(index):
    row = sample_nq_corpus.iloc[index]
        # i=i+1
    print("Processing ",index)
    example_id = row['example_id']
    document_title = 'document_title'#row['document_title']
    document_url = row['document_url']
    document_text = row['document_text'] # avail in simplified nq

    splitter = splitter_arg
    text_chunks = langchain_split.split_input(splitter, None, document_text)
    # text_chunks = [document_text]

    for i,chunk in enumerate(text_chunks):
        chunk_text_dense_embedding = model.encode(chunk) #replace this with document_text (from simplified nq)
        #When using ELSER document_text_embedding is populated through inference pipeline
        doc_elser ={
                        "document_text": chunk,
                        "document_title" : f'{i}_{document_url}',
                        "example_id": f'{example_id}',
                        "document_url": document_url,
            }
        
        doc_knn ={
                        "document_text": chunk,
                        "document_title" : document_title,
                        "example_id": example_id,
                        "document_url": document_url,
                        "document_text_dense_embedding": chunk_text_dense_embedding
                }



## Example Index name
index_name = args.index_name

# Create Index 
with open(args.index_mapping, 'r') as file:
    mapping = file.read().rstrip()
delete_index(index_name)
create_index(index_name,mapping)

## Define folder name 
sample_nq_path = file_path#"./nq-train-sample-250.jsonl"
sample_nq_corpus = pd.read_json(sample_nq_path, lines=True)


# source ="Google NQ"
# index_data(sample_nq_corpus, source, index_name_elser, index_name_knn)

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(index_data, range(100)))
