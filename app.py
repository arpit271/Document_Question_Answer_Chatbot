"""
Multi-PDF Chatbot using Azure OpenAI
Author: Arpit, kumararpit0077@gmail.com

This app allows users to upload multiple PDFs and ask questions based on their content
using Azure OpenAI embeddings and chat models with LangChain and FAISS vector search.
"""

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_pdf_text(pdf_docs):
    """
        Extracts and returns text from uploaded PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
        Splits large text into smaller overlapping chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """
        Creates and stores FAISS vector embeddings using Azure OpenAI.
    """
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment_name="gpt-4o-mini"  # Replace with your Azure embedding deployment
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """
        Builds and returns the QA chain using Azure Chat model.
    """

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, just say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment_name="gpt-4o-mini",  # Replace with your Azure chat deployment
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
    llm=model,
    retriever=FAISS.load_local("faiss_index", AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment="text-embedding-ada-002"
    )).as_retriever()
)


def user_input(user_question):
    """
        Handles user query using Azure embeddings and returns chatbot response.
    """
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment="text-embedding-ada-002"
    )

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])


def main():
    """
        Main function to run Streamlit app.
    """
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF Chatbot 📚🤖")

    user_question = st.text_input("Ask a question from your PDFs...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        # st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDF files and click Submit",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete")

        st.write("---")
        # st.image("img/gkj.jpg")
        st.write("AI App created by Arpit")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; 
        background-color: #0E1117; padding: 15px; text-align: center;">
            © Arpit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()