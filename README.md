# Chunking_to_ES
Script to upload data to Elasticsearch with chunking

## Run
python3 indexing.py --doc_num 1 for uploading data with default configuration

available configurable parameters:

'--file_path', type=str, default='./nq-train-sample-250.jsonl', description='the data path of file to be converted to vector embbedings'

'--splitter', type=str, default='HTMLTextSplitter', description='splitter tp split document'

'--input_text', type=str, default=None, description ='input_text to split'

'--ES_CLOUD_ID', type=str, default=None, description='Elastic Search CLOUD_ID'

'--ES_API_KEY', type=str, default=None, description ='Elastic Search API_KEY'

'--ELSER_Model', type=bool, default=True, description ='Set True if you want to use Elser Elastic serach model'

'--index_name', type=str, default='Index1', description ='input ES index_name'

'--index_mapping', type=str, default='./sparse_encoder_1.txt', description ='index_mapping file path'

'--ES_username', type=str, default='elastic', description='ES_username'

'--ES_password', type=str, default='11ilk50GgQJZ17RV7Zu7b2R0', description='ES_password'

'--number_of_docs', type=int, default=None, description ='number of documents to upload'

'--doc_num', type=int, default=None, description ='Number of document to upload'

