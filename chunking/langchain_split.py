# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from pinecone import Pinecone, ServerlessSpec
from tqdm.notebook import tqdm
import langchain_community
# import openai
# from openai import OpenAI
import string
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
import pandas as pd
import string
import argparse
# AI21SemanticTextSplitter
import os
from getpass import getpass
# from langchain_ai21 import AI21SemanticTextSplitter
from bs4 import BeautifulSoup

from chunking.util import scd_utils

# # Your Pinecone API key
# api_key = "97497599-5ea9-4b33-ac6b-9f04930ad9885"
# #openai API Keys
# openai_key = 'sk-fGVWLsCUvDN7zsMcyHaiT3BlbkFJuIwO1ONZRbxYtcsnXBGMT'

parser = argparse.ArgumentParser(description='SCD')
parser.add_argument('--file_path', type=str, default='./sample-files/SCD-Tej.txt',
                    help='the data path of file to be converted to vector embbedings')
parser.add_argument('--splitter', type=str, default='simpleCharacterTextSplitter, simpleRecursiveCharacterTextSplitter, ai12_default_splitter, ai12_chunk_splitter',
                    help='splitter tp split document')
parser.add_argument('--input_text', type=str, default=None,
                    help='input_text to split')


args = parser.parse_args()

splitter_arg = args.splitter
input_text = args.input_text
file_path = args.file_path

try:
    file_name = file_path.split('/')[-1]
except Exception as e:
    print("An error occurred:", e)

def setup_pinecone(index_name):
    """
    Setup Pinecone with the provided API key and index name.
    
    Parameters:
    index_name (str): The name of the Pinecone index to use.
    
    Returns:
    Pinecone Index object.
    """
    if "PINECONE_API_KEY" not in os.environ:
        os.environ["PINECONE_API_KEY"] = getpass("Enter PINECONE_API_KEY: ")
    
    # Initialize Pinecone client
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"])
    index = pinecone.Index(name=index_name)
    
    return index

def setup_openai():
    """
    Setup OpenAI with the provided API key.
    
    Returns:
    OpenAI client object.
    """
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass("Enter OPENAI_API_KEY: ")
    
    # Initialize OpenAI client
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    return openai

def setup_ai21():
    """
    Setup AI21 with the provided API key.
    """
    if "AI21_API_KEY" not in os.environ:
        os.environ["AI21_API_KEY"] = getpass("Enter AI21_API_KEY: ")#uxOGMyWE6LuW46ZNURJP8mIlq2zTLVjN
    
    # # Initialize AI21 client
    # ai21.api_key = os.environ["AI21_API_KEY"]


def get_simple_processed_text(file_path):
    doc_type = file_path.split('.')[-1]

    if doc_type == scd_utils.DocumentType.txt.name:
        loader = TextLoader(file_path)
        text = loader.load()
        # text_page_content = text[0].page_content

    if doc_type == scd_utils.DocumentType.pdf.name:
        loader = UnstructuredPDFLoader(file_path)
        text = loader.load()
    text_processed = text[0].page_content

    # Explicitly convert text to string
    text_processed = str(text_processed)
    return text_processed

def simpleRecursiveCharacterTextSplitter(text_processed):
    # Create a RecursiveCharacterTextSplitter instance
    recursive_character_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],
            chunk_size=750,
            chunk_overlap=50)
    text_chunks = recursive_character_splitter.split_text(text_processed)
    return text_chunks

def HTMLTextSplitter(text_processed):
    # splitter = HTMLSplitter()

    # Split the HTML content
    html_chunks = parse_and_split_html(text_processed)
    return html_chunks

# Function to parse HTML and split into chunks
def parse_and_split_html(html_content, chunk_size=100):
    soup = BeautifulSoup(html_content.replace('<\/','</'), 'html.parser')
    texts = soup.get_text(separator='\n######\n')
    
    # Split text into chunks
    words = texts.split('\n######\n')
    # chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    chunks = [word for word in words if (word != ' ')]
    # previous_word_append_flag = False
    # for i,word in enumerate(words): 
    #     if (word != ' '):
    #         if len(word.split(' '))<5:
    #             chunk += word
    #             previous_word_append_flag = True
    #         else:
    #             chunk = word  
    #             previous_word_append_flag = False
    #         if previous_word_append_flag != True:
    #             chunks.append(chunk)

    return chunks

def simpleCharacterTextSplitter(text_processed):
    character_splitter = CharacterTextSplitter(
        separator='\n\n',
        chunk_size=750,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    text_chunks = character_splitter.split_text(text_processed)
    return text_chunks

def split_text_ai12(text, max_length=100000):
    """
    Split text into chunks of a specified maximum length.
    
    Parameters:
    text (str): The text to split.
    max_length (int): The maximum length of each chunk.
    
    Returns:
    list: A list of text chunks.
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def ai12_chunk_splitter(text_processed,chunk_size=1000):
    """
    Process the text using AI21 API, ensuring each chunk is within the character limit.
    
    Parameters:
    text (str): The text to process.
    
    Returns:
    list: A list of responses from the AI21 API.
    """
    setup_ai21()
    chunks = split_text_ai12(text_processed)
    responses = []
    semantic_text_splitter = AI21SemanticTextSplitter(chunk_size)

    for chunk in chunks:
        if len(chunk)<30:
            responses += chunk
            return responses

        try:
            response = semantic_text_splitter.split_text(chunk)
            responses += response
        except AssertionError as e:
            print(f"Error processing chunk: {e}")
    
    return responses

def ai12_default_splitter(text_processed):
    setup_ai21()
    semantic_text_splitter = AI21SemanticTextSplitter()

    chunks = split_text_ai12(text_processed)
    responses = []

    for chunk in chunks:
        if len(chunk)<30:
            responses += chunk
            return responses

        try:
            response = semantic_text_splitter.split_text_to_documents(chunk)
            responses += response
        except AssertionError as e:
            print(f"Error processing chunk: {e}")
    
    return responses

    # print(f"The text has been split into {len(documents)} Documents.")
    # for doc in documents:
    #     print(f"type: {doc.metadata['source_type']}")
    #     print(f"text: {doc.page_content}")
    #     print("====")

# Use the splitter's method to split text into chunks
# text_chunks = splitter.split_text(text_processed)
# text_chunks = semantic_text_splitter.split_text(text_processed)
# text_chunks = ai12_split_to_docs(text_processed)
# print(text_chunks)

def split_doc(splitter, file_path):
    splitter = splitter.strip()
    print(f"splitter : {splitter}")
    impl = {
        "ai12_default_splitter": ai12_default_splitter,
        "ai12_chunk_splitter": ai12_chunk_splitter,
        "simpleCharacterTextSplitter": simpleCharacterTextSplitter,
        "simpleRecursiveCharacterTextSplitter": simpleRecursiveCharacterTextSplitter,
        "HTMLTextSplitter":HTMLTextSplitter,
    }[splitter]
    text_processed = get_simple_processed_text(file_path)
    response = impl(text_processed)
    return response

def split_text(splitter, text):
    splitter = splitter.strip()
    print(f"splitter : {splitter}")
    impl = {
        "ai12_default_splitter": ai12_default_splitter,
        "ai12_chunk_splitter": ai12_chunk_splitter,
        "simpleCharacterTextSplitter": simpleCharacterTextSplitter,
        "simpleRecursiveCharacterTextSplitter": simpleRecursiveCharacterTextSplitter,
        "HTMLTextSplitter":HTMLTextSplitter,
    }[splitter]
    response = impl(text)
    return response

def split_input(splitter, file_path, text):
    if file_path != None:
        return split_doc(splitter, file_path) 
    elif text != None:
        return split_text(splitter,text) 
    else:
        return "Please provide Document or Text"
    

# Function to remove punctuation and new lines
# Move the func to utils.py
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).replace('\n', ' ')

# Function to get the embeddings of the text using OpenAI text-embedding-ada-002 model
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   embedding = client.embeddings.create(input=[text], model=model)
   return embedding.data[0].embedding
   
# Assuming df is your DataFrame and 'text_preprocessed' is the column with preprocessed texts
# Note: This operation might take some time depending on the number of texts due to API response times
# df['embedding'] = df['text_preprocessed'].apply(get_embedding)
# df['embedding'] = [get_embedding(text) for text in tqdm(df['text_preprocessed'])]


def upsert_vector_embeddings_to_pinecone(text_embeddings):
    for text in tqdm(df['text_preprocessed']):
        embedding = get_embedding(text)
        index.upsert(
        vectors=[
            {
                "id": f"vec{i}",
                "values": embedding,
                "metadata": {"genre": "machine learning","text": text}
            }
        ],
        namespace= namespace
)

def write_to_csv(file_path,df):
    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    splitter_arr = splitter_arg.split(',')
    for splitter in splitter_arr:
        text_chunks = split_input(splitter, file_path, input_text)
        df = pd.DataFrame(text_chunks)
        write_to_csv(f"./chunked-files/langchain_{splitter}.csv",df)

        # # Set pandas options to display the entire text
        # pd.set_option('display.max_colwidth', None)

        # print(df)
