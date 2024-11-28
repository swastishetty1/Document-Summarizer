import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from io import BytesIO
from typing import List
import re
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, AIMessage

# Cache parsing of PDFs
@st.cache_data
def get_pdf_text(pdf_files: List[BytesIO]) -> str:
    """Extracts and cleans text from uploaded PDFs."""
    text = ""
    for pdf_file in pdf_files:
        pdf = PdfReader(pdf_file)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", page_text)
                page_text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page_text.strip())
                page_text = re.sub(r"\n\s*\n", "\n\n", page_text)
                text += page_text
    return text

# Cache splitting text into chunks
@st.cache_data
def get_chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
    """Splits text into manageable chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Cache creation of FAISS vector store
@st.cache_resource
def get_vector_store(text_chunks: List[str]) -> FAISS:
    """Creates a FAISS vector store from text chunks."""
    embeddings = OCIGenAIEmbeddings(model_id="cohere.embed-english-light-v2.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaweuxa6ovnhihlbpolrh3jrdpasnnukjd5x5slxcekzwsdigsayza")
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def get_conversation_chain(vector_store: FAISS):
    """Sets up a conversational chain with Oracle Generative AI."""
    llm = ChatOCIGenAI(model_id="meta.llama-3.1-405b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaweuxa6ovnhihlbpolrh3jrdpasnnukjd5x5slxcekzwsdigsayza",
    model_kwargs={"temperature": 0, "max_tokens": 4000})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_user_input(question: str):
    """Handles user questions, displays the chat history, and retrieves answers."""
    user_message = HumanMessage(content=question)
    st.session_state.chat_history.append(user_message)
    response = st.session_state.conversation({"question": question})
    assistant_message = AIMessage(content=response["answer"])
    st.session_state.chat_history.append(assistant_message)

    return response["answer"]

def main():
    # Configure Streamlit app
    st.set_page_config(page_title="Chat with Your PDFs", page_icon=":books:")
    st.title("Chat with Your PDFs :books:")

    # Initialize session state variables
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload your PDFs here:")
        uploaded_files = st.file_uploader("Choose PDF Files", type=["pdf"], accept_multiple_files=True)

    # Reset and process new uploads
    if uploaded_files and uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.chat_history = []
        st.session_state.conversation = None
        st.session_state.vector_store = None

        # Extract text, chunk it, and create FAISS vector store
        pdf_text = get_pdf_text(uploaded_files)
        text_chunks = get_chunk_text(pdf_text)
        st.session_state.vector_store = get_vector_store(text_chunks)
        st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
        st.success("PDFs processed successfully. Ready for Q&A!")

    # Interactive chat interface
    if st.session_state.conversation:
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").write(message.content)

        # User input for new questions
        if question := st.chat_input("Ask a question about the document:"):
            # Add user message to chat history
            user_message = HumanMessage(content=question)
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.write(question)

            # Use a spinner while the bot processes the response
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({
                    "question": question,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]

            # Add assistant message to chat history
            assistant_message = AIMessage(content=answer)
            st.session_state.chat_history.append(assistant_message)
            with st.chat_message("assistant"):
                st.write(answer)

if __name__ == "__main__":
    main()
