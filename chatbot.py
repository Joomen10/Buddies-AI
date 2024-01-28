
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import tiktoken
from langchain.callbacks import get_openai_callback
import openai
import img2pdf
import PyPDF2



from jpg2pdf import process_jpg
import io

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('MY_API_KEY')

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        print("No chunks were created from the text.")
        return None
    
    embeddings = OpenAIEmbeddings()
    if chunks and embeddings:
        knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase
    
                
def main():
    st.title("pdf")
    uploaded_file = st.file_uploader("Upload your JPG file", type='jpg')

    if uploaded_file is not None:
        #jpg_bytes = io.BytesIO(uploaded_file.getvalue())
        #pdf = convert_jpg_to_pdf(jpg_bytes)
        #jpg_bytes = io.BytesIO(uploaded_file.read())
        #pdf = process_jpg(jpg_bytes)
        pdf = img2pdf.convert(uploaded_file.read())


        
        if pdf is not None:

            reserve_pdf_on_memory = io.BytesIO(pdf)
            pdf_reader = PdfReader(reserve_pdf_on_memory)

            #pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                print(text)
            
            knowledgeBase = process_text(text)
            if knowledgeBase is None:
                st.write("No text was found in the uploaded file. Please make sure you're uploading a file that contains readable text.")
                st.stop()
                
            
            query = st.text_input('Ask a question to the PDF')
            cancel_button = st.button('Cancel')
            
            if cancel_button:
                st.stop()
            
            if query:
                docs = knowledgeBase.similarity_search(query)
                chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)
                            
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)
                    
                st.write(response)
        else:
            st.error("Failed to convert the image to PDF.")
            
if __name__ == "__main__":
    main()
