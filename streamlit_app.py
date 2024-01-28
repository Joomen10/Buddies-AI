import requests
import streamlit as st
import os
import tiktoken
import openai

from PIL import Image
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from text2voice import text_to_speech


st.set_page_config(page_title="emPATHy", page_icon="🤖", layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

image_path = "images/brand_logo2.png" 

with st.container():
    text_column, image_column = st.columns((2,1))
    with text_column:
        st.subheader("")
        st.title("Buddies AI for Elders")
        st.write(
        "We provide personal AI assistence for elders to navigate the fast-changing world!"
        )
    with image_column:
        st.image(image_path, use_column_width=True)

# ---- INTRO SECTION ----
with st.container():
    st.write("---")
    st.header("What Buddies AI Can Do For You")
    st.write("##")
    st.write("""
    Experience effortless technology with Buddies AI, bridging the elderly and the digital world. Our chatbot empowers those challenged by technology, offering:

    """)
    
    st.info("1. An easy-to-use solution for adapting to the fast-paced, technology-driven world, perfect for those not familiar with AI and modern tech.")

    st.info("2. Personalized assistance in understanding and using new technological tools, making life simpler and more connected.")

    st.info("3. Expertise in breaking down long, complex documents into manageable, easy-to-understand formats, ideal for those who struggle with extensive reading.")

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("Explore Our Chatbot!")

    st.subheader("1. Simple Document Summaries")           
    st.info("""
                Easily turn long, complex documents into short, easy-to-understand summaries with just a click. Perfect for getting the gist quickly!
                """)

    st.subheader("2. Key Information Highlighting")
    st.info("""
                Our chatbot helps you find the most important parts of any document, saving you time and hassle in sorting through pages of text.
                """)

    st.subheader("3. Answers to Your Questions")
    st.info("""
                Got a question about your document? Just ask, and our AI will provide clear, straightforward answers using the document's information.
            """)

    st.subheader("4. Spoken Responses for Better Accessibility")
    st.info("""
                Our chatbot can read out information to you, making it easier to understand and access documents, especially if reading text is challenging.
            """)

        
    #st.write(""")Below is our Beta version of JPG summarizer service. 

      #  """)
    image_column, text_column = st.columns((1, 2))


load_dotenv()

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
   
    os.environ["OPENAI_API_KEY"] = os.getenv('MY_API_KEY')
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

# ---- Buddines AI ----    
with st.container():
    st.write("---")
    def main():
        st.title("Upload your PDF Document")
        pdf = st.file_uploader('', type='pdf')
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
                chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)
                            
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)
                    
                st.write(response)
                text_to_speech(response)
  
                

            
if __name__ == "__main__":
    main()

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Inquiry Form: Let's Explore Together")
    st.write("##")

    contact_form = """
    <form action="https://formsubmit.co/jjjr0617@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

