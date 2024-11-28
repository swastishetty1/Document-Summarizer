from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from typing import List
import base64
from io import BytesIO

app = FastAPI()
class FileInput(BaseModel):
    file_name: str
    content: str 

class FilesRequest(BaseModel):
    files: List[FileInput]

def pdf_summariser(vectorstore):
    query = "Summarize the content of the uploaded file"

    if query:
        # Retrieve relevant chunks from the vector store
        docs = vectorstore.similarity_search(query)
    
    # Load summarization chain using ChatOCIGenAI
    chain = load_qa_chain(ChatOCIGenAI(model_id="meta.llama-3.1-405b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaweuxa6ovnhihlbpolrh3jrdpasnnukjd5x5slxcekzwsdigsayza",
    model_kwargs={"temperature": 0, "max_tokens": 4000}))
    summary = chain.run(input_documents=docs, question=query)
    return summary

def vector_store(text_chunks):
    embeddings = OCIGenAIEmbeddings(model_id="cohere.embed-english-light-v2.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaweuxa6ovnhihlbpolrh3jrdpasnnukjd5x5slxcekzwsdigsayza")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore    

def get_text_chunks(raw_text):
    print("Raw text:")
    print(raw_text)
    # if not isinstance(raw_text, str):
    #     raise ValueError("Input to text splitter must be a string.")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(text=raw_text)
    if not text_chunks:
        raise ValueError("Text splitting failed; no valid chunks created.")
    return text_chunks

def process_pdf_text(pdf_files):    
    if pdf_files is not None:
        raw_text = ""
        for file in pdf_files:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()

        return raw_text

@app.post("/summarize/")
async def summarize_files(files_request: FilesRequest):

    try:

        decoded_files = [
            BytesIO(base64.b64decode(file.content)) for file in files_request.files
        ]
        print(decoded_files)
        # Extract text from uploaded files
        raw_text = process_pdf_text(decoded_files)
        # print(raw_text)
        # Split the text into chunks
        text_chunks = get_text_chunks(raw_text)
        print(text_chunks)
        # Create a vector store
        vectorstore = vector_store(text_chunks)
        print(vectorstore)
        # Summarize using the vector store
        summary = pdf_summariser(vectorstore)

        return {"summary": summary}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "File Summarizer API!"}