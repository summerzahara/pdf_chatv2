from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def process_pdf(pdf):
    #Load
    pdf_reader = PdfReader(pdf)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()

    #Split
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)

    # Embed
    embeddings = OpenAIEmbeddings()
    embeddings.embed_documents(chunks)

    # Store
    db = FAISS.from_texts(chunks, embeddings)
    return db

def retrieve_data(db, query):
    # retriever = db.as_retriever()
    # response = retriever.get_relevant_documents(query)
    response = db.similarity_search(query)
    return response[0].page_content