
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

load_dotenv()

def process_text(text):
    #tokenizer = tiktoken.get_encoding("cl100k_base")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    #token_encoding = tiktoken.get_encoding(text)
    chunks = text_splitter.split_text(text)
    #tokens = davinci_tokenizer.tokenize(chunks)
    #embeddings = OpenAIEmbeddings(model="davinci-002")
    #knowledgeBase = FAISS.from_texts(chunks, embeddings)

    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key = "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"


    
    comparison_dict = {}
    embeddings_list = []

    #for example_string in chunks:
        # Removed the update to comparison_dict as it wasn't defined in your original code
 
        #embeddings_list.append(OpenAIEmbeddings(response))  # Assuming OpenAIEmbeddings can process the response
    #tokenizer = tiktoken.get_encoding(encoding_name)
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    #knowledgeBase = FAISS.from_texts(chunks, embeddings_list)  # Assuming FAISS.from_texts can handle the list of embeddings
    return knowledgeBase
   

def get_encoding_lengths(example_string: str) -> None:
    results = {}
    for encoding_name in ["gpt2", "p50k_base", "cl100k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        num_tokens = len(token_integers)
        results[encoding_name] = num_tokens
    return {example_string: results}


        
                
def main():
    st.title("pdf")
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        knowledgeBase = process_text(text)
        
        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            docs = knowledgeBase.similarity_search(query)
            llm = OpenAI()
            chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
                        
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
            st.write(response)
            
if __name__ == "__main__":
    main()
