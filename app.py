import streamlit as st
from utils.loader import load_and_split
from utils.vectorestore import create_vectorstore
from utils.chain import answer_question
from io import BytesIO


def main():
    st.title('Advanced RAG Q&A System')
    input_type = st.selectbox(
        "Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    input_data = None
    if input_type == "Link":
        number_input = st.number_input(
            "Enter the number of Links", min_value=1, max_value=20, step=1, value=1)
        input_data = []
        for i in range(int(number_input)):
            url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
            if url:
                input_data.append(url)
    elif input_type == "Text":
        input_data = st.text_area("Enter the text", height=200)
    elif input_type == "PDF":
        input_data = st.file_uploader("Upload a PDF file", type=["pdf"])
    elif input_type == "TXT":
        input_data = st.file_uploader("Upload a text file", type=["txt"])
    elif input_type == "DOCX":
        input_data = st.file_uploader(
            "Upload a DOCX file", type=["docx", "doc"])

    if st.button("Proceed"):
        if not input_data:
            st.error("Please provide valid input (e.g., file, text, or URLs).")
        else:
            try:
                with st.spinner("Processing input..."):
                    if input_type == "Text":
                        # Pass text directly
                        docs = load_and_split(input_type, input_data)
                    elif input_type == "Link":
                        # Pass list of URLs
                        docs = load_and_split(input_type, input_data)
                    else:
                        # Wrap file in BytesIO for consistency
                        docs = load_and_split(input_type, input_data)
                    vectorstore = create_vectorstore(docs)
                    st.session_state['vectorstore'] = vectorstore
                    st.success("Input processed successfully!")
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit", key="submit_with_query") and query.strip():
            with st.spinner("Generating answer..."):
                try:
                    answer = answer_question(
                        st.session_state["vectorstore"], query)
                    st.write("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        elif st.button("Submit", key="submit_no_query") and not query.strip():
            st.warning("Please enter a question.")


if __name__ == '__main__':
    main()
