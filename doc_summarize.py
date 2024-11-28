import os
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from PyPDF2 import PdfReader

# Function for summarizing the PDF files
def multi_pdf_summariser(vectorstore):
    query = "Summarize the content of the uploaded "

    if query:
        # Retrieve relevant chunks from the vector store
        docs = vectorstore.similarity_search(query)
    
    # Load summarization chain using ChatOCIGenAI
    chain = load_qa_chain(ChatOCIGenAI(model_id="meta.llama-3.1-405b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaweuxa6ovnhihlbpolrh3jrdpasnnukjd5x5slxcekzwsdigsayza",
    model_kwargs={"temperature": 0, "max_tokens": 4000}), chain_type="stuff")
    summary = chain.run(input_documents=docs, question=query)
    st.write(summary)

# Function to embed text chunks and store them in FAISS vector DB
def vector_store(text_chunks):
    # Embed the texts using OCIGenAIEmbeddings
    embeddings = OCIGenAIEmbeddings(model_id="cohere.embed-english-light-v2.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaweuxa6ovnhihlbpolrh3jrdpasnnukjd5x5slxcekzwsdigsayza")

    # Create FAISS vector store
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    # Summarize using the vector store
    multi_pdf_summariser(vectorstore)

# Function to split text into chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(text=raw_text)
    vector_store(text_chunks)

# Function to extract text from uploaded PDFs
def process_pdf_text(pdf_files):
    if pdf_files is not None:
        raw_text = ""
        for file in pdf_files:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
        
        # Process extracted text into chunks
        get_text_chunks(raw_text)

# Main Streamlit App
def main():
    # Configure Streamlit page
    st.set_page_config(page_title="PDF Summarizer", layout="wide")

    # Set up page layout
    st.title("PDF SUMMARIZER")
    st.write("Summarize your confidential PDFs in seconds")
    st.divider()

    # Add custom styling to the sidebar
    st.markdown(
        """
        <style>
            .st-emotion-cache-vk3wp9.eczjsme11 {
                background-color: #EAEAEA;  /* Sidebar Background */
                color: #333333;
            }
            .st-emotion-cache-13ejsyy.ef3psqc12 {
                background-color: #008080;  /* Button Background */
                color: #EAEAEA;
            }
            .st-emotion-cache-taue2i.e1b2p2ww15 {
                background-color: #F5F5F5;  /* Upload Box Background */
                border: 1px solid #008080;
            }
            .st-emotion-cache-10trblm.e1nzilvr1 {
                color: #008080;  /* Main Title */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Summarizer")
        
        # File uploader for PDFs
        pdf_files = st.file_uploader(
            "Upload your PDF Document", type="pdf", accept_multiple_files=True
        )

        # Button to trigger summarization
        submit = st.button("Upload")

    if submit:
        st.subheader("Summary of the Files:")
        with st.spinner("Getting your PDF Summary Ready..."):
            process_pdf_text(pdf_files)

if __name__ == "__main__":
    main()
