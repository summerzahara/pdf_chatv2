import streamlit as st
from llm_helper import process_pdf, retrieve_from_db

def generate_response(query):
        response = st.session_state.chat({"question": query})
        st.session_state.chat_history = response['chat_history']

        for i, element in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.chat_message("user").write(element.content)
            else:
                st.chat_message("assistant").write(element.content)
        

def main():
    st.title("PDF Chat")
    # Add ability to upload pdf

    # Set Session State
    if 'chat' not in st.session_state:
        st.session_state.chat = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
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

    user_question = st.chat_input("What is your question?")
    if user_question:
        generate_response(user_question)

        



if __name__ == "__main__":
    main()