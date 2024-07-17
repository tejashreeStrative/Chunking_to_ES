from llama_index.core.node_parser import SimpleFileNodeParser, HierarchicalNodeParser, SentenceWindowNodeParser, SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file.flat.base import FlatReader
from pathlib import Path
import argparse
import fitz  # PyMuPDF
import pandas as pd
import openai
import os
from getpass import getpass

parser = argparse.ArgumentParser(description='SCD')
parser.add_argument('--file_path', type=str, default='./sample-files/SCD-Tej.txt',
                    help='the data path of file to be converted to vector embbedings')
parser.add_argument('--splitter', type=str, default='simpleFileNodeParser, simpleTokenTextSplitter, simpleSentenceSplitter, simpleSemanticSplitter, simpleSentenceWindowSplitter, simpleHierarchicalSplitter',
                    help='splitter tp split document')


args = parser.parse_args()

splitter_arg =args.splitter
file_path = args.file_path
try:
    file_name = file_path.split('/')[-1]
except Exception as e:
    print("An error occurred:", e)

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

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using PyMuPDF.
    
    Parameters:
    file_path (str): The path to the PDF file.
    
    Returns:
    str: Extracted text content.
    """
    text = ""
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"An error occurred while extracting text from the PDF: {e}")
    return text

def simpleFileNodeParser(file_path):
    """
    Process the PDF file using the llama_index library.
    
    Parameters:
    file_path (str): The path to the PDF file.
    
    Returns:
    list: A list of nodes extracted from the PDF.
    """
    try:
        # Initialize the PDF loader
        extracted_text = FlatReader().load_data(Path(file_path))
        # print(extracted_text)
        # Parse the documents into nodes
        parser = SimpleFileNodeParser()
        md_nodes = parser.get_nodes_from_documents(extracted_text)
        
        return md_nodes
    except Exception as e:
        print(f"An error occurred while processing the PDF with llama_index: {e}")
        return []

    
def simpleSentenceSplitter(file_path):
    documents = FlatReader().load_data(Path(file_path))

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def simpleTokenTextSplitter(file_path):
    documents = FlatReader().load_data(Path(file_path))

    splitter = TokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separator=" ",
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def simpleSemanticSplitter(file_path):
    setup_openai()
    documents = FlatReader().load_data(Path(file_path))

    embed_model = OpenAIEmbedding()
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def simpleSentenceWindowSplitter(file_path):
    documents = FlatReader().load_data(Path(file_path))

    splitter = SentenceWindowNodeParser.from_defaults(
        # how many sentences on either side to capture
        window_size=3,
        # the metadata key that holds the window of surrounding sentences
        window_metadata_key="window",
        # the metadata key that holds the original sentence
        original_text_metadata_key="original_sentence",
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


def simpleHierarchicalSplitter(file_path):
    documents = FlatReader().load_data(Path(file_path))

    splitter = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128]
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def get_simple_processed_text(file_path):
    doc_type = file_path.split('.')[-1]

    if doc_type == scd_utils.DocumentType.txt.name:
        loader = TextLoader(file_path)
        text = loader.load()

    if doc_type == scd_utils.DocumentType.pdf.name:
        loader = UnstructuredPDFLoader(file_path)
        text = loader.load()
    text_processed = text[0].page_content

    # Explicitly convert text to string
    text_processed = str(text_processed)
    return text_processed

def split_doc(splitter, file_path):
    splitter = splitter.strip()
    print(f"splitter : {splitter}")
    text_processed = get_simple_processed_text(file_path)

    impl = {
        "simpleFileNodeParser": simpleFileNodeParser,
        "simpleTokenTextSplitter": simpleTokenTextSplitter,
        "simpleSentenceSplitter": simpleSentenceSplitter,
        "simpleSemanticSplitter": simpleSemanticSplitter,
        "simpleSentenceWindowSplitter": simpleSentenceWindowSplitter, 
        "simpleHierarchicalSplitter": simpleHierarchicalSplitter
    }[splitter]
    response = impl(file_path)
    return response

def write_to_csv(file_path,df):
    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

if  __name__ == "__main__":
    
    splitter_arr = splitter_arg.split(',')
    for splitter in splitter_arr:
        md_nodes = split_doc(splitter, file_path)
        df = pd.DataFrame(md_nodes)
        write_to_csv(f"./chunked-files/llama_{splitter}.csv",df)

        # Set pandas options to display the entire text
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.width', None)
        # print(df.head())
