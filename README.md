# LikeLion

# Buddies_AI

Buddies_AI is a website coded with python and streamlit that allows the elders to input a pdf file and interact with an AI Chatbot with possible questions. If needed, users can activate a voice-reading feature for the answer that the OpenAI assistant responded. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Buddies_AI.

```bash
pip install streamlit
pip install dotenv
pip install langchain
pip install tiktoken
pip install PyPDF2
pip install text2voice
pip install gTTS
pip install python
pip install openai 
```

## Usage

```python

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
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
