import os
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document  # For Word documents

# Function for summarizing the files
def multi_file_summariser(vectorstore):
    query = "Summarize the content of the uploaded file in 4-5 sentences. Start answer with File Name."

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
    multi_file_summariser(vectorstore)

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

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    raw_text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
    return raw_text

# Function to extract text from Word files
def extract_text_from_word(file):
    doc = Document(file)
    raw_text = ""
    for paragraph in doc.paragraphs:
        raw_text += paragraph.text + "\n"
    return raw_text

# Function to extract text from text files
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Generalized function to process all file types
def process_uploaded_files(files):
    raw_text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            raw_text += extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            raw_text += extract_text_from_word(file)
        elif file.name.endswith(".txt"):
            raw_text += extract_text_from_txt(file)
        else:
            st.warning(f"Unsupported file type: {file.name}")
    if raw_text:
        get_text_chunks(raw_text)

# Main Streamlit App
def main():
    # Configure Streamlit page
    st.set_page_config(page_title="File Summarizer", layout="wide")

    # Set up page layout
    st.title("FILE SUMMARIZER")
    st.write("Summarize your confidential documents in seconds")
    st.divider()

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Summarizer")
        # File uploader for PDFs, Word docs, and text files
        uploaded_files = st.file_uploader(
            "Upload your Document", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=False
        )

        # Button to trigger summarization
        submit = st.button("Upload")

    if submit:
        st.subheader("Summary of the Files:")
        with st.spinner("Getting your file summary ready..."):
            process_uploaded_files(uploaded_files)

if __name__ == "__main__":
    main()
