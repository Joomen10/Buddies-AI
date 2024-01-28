import requests
import streamlit as st

from PIL import Image

from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="emPATHy", page_icon="🤖", layout="wide")


# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()



# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
# lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
# img_contact_form = Image.open("images/yt_contact_form.png")
# img_lottie_animation = Image.open("images/yt_lottie_animation.png")

image_path = "images/brand_logo2.png" 
# ---- HEADER SECTION ----
with st.container():
    text_column, image_column = st.columns((2,1))
    with text_column:
        st.subheader("")
        st.title("Buddies AI for Elderly")
        st.write(
        "We provide personal AI assistence for elders to navigate the fast-changing world!"
        )
    with image_column:
        st.image(image_path, use_column_width=True)

# ---- INTRO SECTION ----
with st.container():
    st.write("---")
    st.header("What Our ChatBot Does and Why Us?")
    st.write("##")
    st.write("""
    Buddies AI ChatBot aims to create a seamless AI environment for the global community.
    With our ChatBot, we provide servieces for people, especially elders, who:
    """)
    
    st.info("1. are looking for a way to live their lives in this fact-changing world much easier and more convinient.")

    st.info("2. are struggling with the new technology that has been utilized in many different places without any firm guide.")

    st.info("3. Tare working with tons of words and languages with low vision - 'there has to be a way to listen to them.'")

# with st.container():
    # st.write("---")
    # left_column, right_column = st.columns(2)
    # with left_column:
    #     st.header("What Our Company Does and Why Us?")
    #     st.write("##")
    #     st.write(
    #         """
    #         Buddies AI aims to create a seamless AI environment for the global community.

    #         With our company, we provide servieces for people, especially elders, who:
    #         - are looking for a way to live their lives in this fact-changing world much easier and more convinient.
    #         - are struggling with the new technology that has been utilized in many different places without any firm guide.
    #         - are working with tons of words and languages with low vision - "there has to be a way to listen to them."
 
    #         """
    #     )
    # with right_column:
    #     st_lottie(lottie_coding, height=300, key="coding")

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("Our Technologies")
    st.write(
        """
            Discover Our Technologies:
        """)

    st.subheader("1. PDF Summarizer")           
    st.info("""
                Simplifies complex information, making it easy for elders to grasp key insights.
                """)
    
    st.subheader("2. Voice-Activated Assistants")
    st.info("""
                Elders can control their environment using intuitive voice commands, 
                from setting reminders to controlling smart home devices.
                """)
    
    st.subheader("3. Health Monitoring Wearables")
    st.info("""
                Well-being with wearable devices monitor vital signs, providing real-time health information for both elders and their caregivers.
            """)
    st.subheader("4. Medication Reminder System")
    st.info("""
                Ensure medication adherence with the smart reminder system, with timely alerts and peace of mind for both elders and their families.
            """)
    
 #    st.info("""1. PDF Summarizer:
 #               Revolutionizing document accessibility, our PDF reader simplifies complex information, making it easy for elders to grasp key insights effortlessly.
 #               """)
 #   st.info("""2. Voice-Activated Assistants:
 #               Empower elders with the ability to control their environment using intuitive voice commands, from setting reminders to controlling smart home devices.
 #               """)
 #   st.info("""3. Health Monitoring Wearables:
 #               Prioritize well-being with wearable devices that monitor vital signs, providing real-time health information for both elders and their caregivers.
 #           """)
 #   st.info("""4. Medication Reminder System:
 #              Ensure medication adherence with our smart reminder system, offering timely alerts and peace of mind for both elders and their families.
 #           """)

    st.write("""Below is our Beta version of JPG summarizer service. 

        """)
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
    os.environ["OPENAI_API_KEY"] = "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"
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def main():
    st.title("Give us what you need to summarize!")
    pdf = st.file_uploader('Upload your image', type='jpg')
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
            chain = load_qa_chain(llm=OpenAI(), chain_type='stuff')
            response = chain.run(input_documents=docs, question=query)
                        
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
            st.write(response)
            
if __name__ == "__main__":
    main()

    # with image_column:
    #     st.image(img_lottie_animation)
    # with text_column:
    #     st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
    #     st.write(
    #         """
    #         Learn how to use Lottie Files in Streamlit!
    #         Animations make our web app more engaging and fun, and Lottie Files are the easiest way to do it!
    #         In this tutorial, I'll show you exactly how to do it
    #         """
    #     )
    #     st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")
# with st.container():
#     image_column, text_column = st.columns((1, 2))
    # with image_column:
    #    st.image(img_contact_form)
    # with text_column:
    #     st.subheader("How To Add A Contact Form To Your Streamlit App")
    #     st.write(
    #         """
    #         Want to add a contact form to your Streamlit website?
    #         In this video, I'm going to show you how to implement a contact form in your Streamlit app using the free service ‘Form Submit’.
    #         """
    #     )
    #     st.markdown("[Watch Video...](https://youtu.be/FOULV9Xij_8)")

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Inquiry Form: Let's Explore Together")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
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

