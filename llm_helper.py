from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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

def retrieve_and_chat(db):
    llm = ChatOpenAI()
    retriever = db.as_retriever()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=True,
    )
    return conversation