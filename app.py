import streamlit as st
from llm_helper import process_pdf, retrieve_data

def main():
    st.title("PDF Chat")
    # Add ability to upload pdf

    # Set Session State
    if 'vector_db' not in st.session_state:
        st.session_state['vector_db'] = None
    
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
            vector_db = process_pdf(my_pdf)
            st.session_state['vector_db'] = vector_db
            st.info("Processed")

    with st.form("Query Form"):
        user_query = st.text_input("Submit a query:")
        submit_query = st.form_submit_button("Query")
        if submit_query:
            response = retrieve_data(st.session_state['vector_db'],user_query)
            st.write(response)


if __name__ == "__main__":
    main()