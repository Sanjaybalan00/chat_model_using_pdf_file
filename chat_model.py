import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
import os


Google_API_Key = 'your_key'

current_dir=os.path.dirname(os.path.abspath(__file__))
persistent_directory=os.path.join(current_dir,'db',"chroma_db")
# Loading PDF texts
def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Splitting the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
    chunks = text_splitter.split_text(text)
    return chunks


# Convert the chunks to embeddings and store them in FAISS
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=Google_API_Key)
    vector_store = Chroma.from_texts(chunks, embedding=embeddings,persist_directory=persistent_directory)


# Setup conversational chain with Gemini model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model = GoogleGenerativeAI(model='gemini-pro', google_api_key=Google_API_Key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Process user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=Google_API_Key)
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])


# Main function to create the Streamlit app layout
def main():
    # Header
    st.markdown('<div class="header">Chat with PDF using Gemini</div>', unsafe_allow_html=True)

    # User interaction for asking questions
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h3>Ask a Question from the PDF Files</h3>', unsafe_allow_html=True)
    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question)

    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Menu:")
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
        if st.button("Submit & Process", key="submit", help="Upload and process the PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_text(pdf_docs)  # Fixed function name
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF processing is complete and ready for Q&A!")
            else:
                st.warning("Please upload at least one PDF file.")
        st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])  # Create two columns
with col1:
    st.write("")  # Just to maintain alignment, can be removed if not needed
    with col2:
        st.image("cat.gif", use_column_width=True)         


if __name__ == "__main__":
    main()
