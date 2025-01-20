# Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader  # For reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable chunks
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For embeddings generation using Google GenAI
import google.generativeai as genai  # For configuring Google GenAI API
from langchain.vectorstores import FAISS  # For managing vector stores
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat models using Google GenAI
from langchain.chains.question_answering.chain import load_qa_chain  # For creating QA chains
from langchain.prompts import PromptTemplate  # For defining custom prompts
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")  # Ensure the Google API key is loaded correctly

# Configure the Google GenAI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    """
    Reads text from a list of uploaded PDF files.
    Args:
        pdf_docs (list): List of uploaded PDF files.
    Returns:
        str: Extracted text from all PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Initialize PDF reader
        for page in pdf_reader.pages:  # Loop through all pages
            text += page.extract_text()  # Append extracted text
    return text


# Function to split large text into smaller chunks for processing
def get_text_chunks(text):
    """
    Splits text into smaller chunks for better processing.
    Args:
        text (str): Large input text.
    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    return chunks


# Function to create and save a vector store for the text chunks
def get_vector_store(text_chunks):
    """
    Creates a vector store from text chunks using Google Generative AI embeddings
    and saves it locally.
    Args:
        text_chunks (list): List of text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create vector store
    vector_store.save_local("faiss_index")  # Save the vector store locally


# Function to create a conversational chain for answering questions
def get_conversational_chain():
    """
    Creates a conversational chain using Google Generative AI.
    Returns:
        chain: A loaded question-answering chain.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". Don't provide a wrong answer.
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)  # Initialize chat model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Define custom prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Load QA chain with the model and prompt
    return chain


# Function to process user input and generate a response
def user_input(user_question):
    """
    Handles user input, retrieves relevant documents from vector store,
    and generates a response using the conversational chain.
    Args:
        user_question (str): User's question.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load vector store
    docs = new_db.similarity_search(user_question)  # Search for similar documents based on the question
    chain = get_conversational_chain()  # Get the QA chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)  # Generate response
    print(response)  # Print response to console
    st.write("Reply: ", response["output_text"])  # Display response in the Streamlit app


# Main function for Streamlit app
def main():
    """
    Main function to run the Streamlit app for interacting with PDFs and the model.
    """
    st.set_page_config("Chat PDF")  # Set Streamlit page configuration
    st.header("Chat with PDF using Gemini💁")  # App header

    user_question = st.text_input("Ask a Question from the PDF Files")  # User input field for questions

    if user_question:
        user_input(user_question)  # Process user input if a question is asked

    with st.sidebar:  # Sidebar for uploading and processing PDFs
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):  # Button to process uploaded PDFs
            with st.spinner("Processing..."):  # Show spinner while processing
                raw_text = get_pdf_text(pdf_docs)  # Extract text from PDFs
                text_chunks = get_text_chunks(raw_text)  # Split text into chunks
                get_vector_store(text_chunks)  # Create vector store from chunks
                st.success("Done")  # Display success message


# Run the app
if __name__ == "__main__":
    main()
