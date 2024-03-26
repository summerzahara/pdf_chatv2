# Project Overview
In this project, I created a PDF Chat App using: python, langchain, streamlit, and OpenAI

[YouTube Tutorial](https://youtu.be/V3PWJ7gmOdM?si=EpOvi_WR3qOf2YXE)


## Getting Started
- Knowledge of python
- IDE
- OpenAI account

## Virtual Environment
```zsh
pip3 install virtualenv
```

```zsh
python3 -m venv <your-env-name>
```

```zsh
source <your-env-name>/bin/activate
```

## Libraries
- I like to create a `requirements.txt` file for installing libraries
```txt
langchain
langchain_community
langchain-openai
faiss-cpu
pypdf
python-dotenv
streamlit
```

```zsh
pip3 install -r requirements.txt
```

## Upload PDF
- In `app.py` write code to allow ability to upload pdf
```python
import streamlit as st

def main():
	st.title("PDF Chat")
	with st.sidebar:
		st.file_uploader(
			"Upload File:",
			type=["pdf"],
		)

if __name__=="__main__":
	main()
```

## 1/ Load
- In order to load the text from the streamlit UploadedFile object (what the file_uploader returns), we need to use the pypdf PDFReader
- We are going to create function in the `llm_helper.py` to read the pdfs

```python
from pypdf import PdfReader

def process_pdf(pdf):
	# Load data
	pdf_reader = PdfReader(pdf)
	text = ""
	for page in pdf_reader.pages:
		text += page.extract_text()
	return text
```

- Then we will import this into `app.py`

```python
import streamlit as st
# Import our function
from llm_helper import process_pdf

def main():
	st.title("PDF Chat")
	with st.sidebar:
		#Make pdf a variable
		my_pdf = st.file_uploader(
			"Upload File:",
			type=["pdf"],
			accept_multiple_files=False,
		)
		# Add a button when ready to read text
		submit = st.button(
			"Submit",
			type="primary",
		)

		# Trigger our function to load pdf to text
		if submit:
			result=process_pdf(pdf)
			st.write(result)
```

## 2/ Split
- Back in `llm_helper.py` we will update our function to split the text into chunks
- This requires the CharacterTextSplitter from langchain

```python
from pypdf import PdfReader
# Import Text Splitter
from langchain_text_splitters import CharacterTextSplitte
 

def process_pdf(pdf):
	# Load
	pdf_reader = PdfReader(pdf)
	text = ""
	for page in pdf_reader.pages:
	text += page.extract_text()

	# Split
	text_splitter = CharacterTextSplitter(
		separator="\n",
		chunk_size=1000,
		chunk_overlap=200,
		length_function=len,
		is_separator_regex=False,
	)
	chunks = text_splitter.split_text(text)
	return chunks
```

- Back in `app.py` we can rerun our app, and now view the chunks that have been created from out PDF text


## 3/ Embed
- To get started with embedding we need to connect to OpenAI
- Create an `.env` file to store API key

```python
OPENAI_API_KEY="sk-1234"
```

- In `llm_helper.py` Use `load_dotenv` to import as environment variable

```python
from dotenv import load_dotenv

load_dotenv()
```

- Now we can start embedding

```python
from langchain_openai import OpenAIEmbeddings


def process_pdf(pdf):
	# Load
	pdf_reader = PdfReader(pdf)
	text = ""
	for page in pdf_reader.pages:
	text += page.extract_text()

	# Split
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
	return embeddings
```

- Back in `app.py` if we rerun our app, we can see that it is now returning the embeddings

## 4/ Store
This walkthrough uses the `FAISS` vector database, which makes use of the Facebook AI Similarity Search (FAISS) library.

```python
from langchain_community.vectorstores import FAISS


def process_pdf(pdf):
	# Load
	pdf_reader = PdfReader(pdf)
	text = ""
	for page in pdf_reader.pages:
	text += page.extract_text()

	# Split
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
```

- Back in `app.py`, when we rerun the app, we should see the vector store

## 5/ Retrieve

- Now we want to set up a retriever to retrieve relevant chunks based on user input. To do this we need to connect to the llm from OpenAI
- Next, we setup our vector db "as retriever"
- After that we need to setup chat memory storage
- To return the chat history, we use a ConversationRetrievalChain from langchain

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def process_pdf(pdf):
	# Load
	pdf_reader = PdfReader(pdf)
	text = ""
	for page in pdf_reader.pages:
	text += page.extract_text()

	# Split
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

	# Retrieve
	llm = ChatOpenAI()
	retriever = db.as_retriever()
	memory = ConversationBufferMemory(
		memory_key='chat_history",
		return_messages=True,
	)
	conversation = ConversationalRetrievalChain.from_llm(
		llm=llm,
		memory=memory,
		retriever=retriever,
		verbose=True,
	)
	return conversation
```


- Back in `app.py`, we will make a few updates
- First, I renamed our output from "result" to "chat"
- Next, we need to set the "session state" for the our chat history so we can access this globally
- Lastly, instead of writing the results to the screen, let's create a 'info' box to tell us when the pdf has finished processing

```python
def main():
	st.title("PDF Chat")
	# Add ability to upload pdf 
	
	# Set Session State
	if 'chat' not in st.session_state:
		st.session_state.chat = None
	
	with st.sidebar:
		my_pdf = st.file_uploader(
			"Upload File:",
			type=['pdf'],
			accept_multiple_files=False,
		)

		# Add button to trigger process
		submit = st.button(
			"Submit",
			type="primary",
		)
  
		# Add logic to trigger process function
		if submit:
			st.session_state.chat = process_pdf(my_pdf)
			st.info("Processed")
```

## 6/ Generate
- Now we need to to generate our chat
- Let's start by creating a user input

```python
user_query = st.chat_input("Enter your query")
```

- Next we need to create a function to generate our chat responses

```python
# Set Session State
	if 'chat' not in st.session_state:
		st.session_state.chat = None
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = None

def generate_response(query):
	response = st.session_state.chat(
		{
			"question": query,
		}
	)
	st.session_state.chat_history = response['chat_history']

	for i, element in enumerate(st.session_state.chat_history):
		if i % 2 == 0:
			st.chat_message("user").write(element.content)
		else:
			st.chat_message("assistant").write(element.content)
```

- With this set up, we can call this function whenever a user submits a query

```python
user_query = st.chat_input("Enter your query")
if user_query:
	generate_response(user_query)
```
