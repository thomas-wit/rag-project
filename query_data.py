import argparse
# from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from openai import OpenAI
import streamlit as st
from config import KEY

client = OpenAI(api_key = KEY)

CHROMA_PATH = "chroma"

##### Run the streamlit app by using the command "streamlit run query_data.py"




PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Call OpenAI API
    response = client.chat.completions.create(model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" or other models
    messages=[
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])

    response_text = response.choices[0].message.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return formatted_response


def main():
    ## Adding a title for the Streamlit app
    st.title("Ask Me Anything")

    # Adding a subtitle
    st.subheader("... as long as it is about Church History...")

    # setup a session state message variable to hold all of the session messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Build a prompt input for the user

    prompt = st.chat_input('Ask Your Church History Question Here')

    # If user hits enter then
    if prompt:
        # Display prompt
        st.chat_message('user').markdown(prompt)
        # store user prompt
        st.session_state.messages.append({'role':'user', 'content':prompt})
        # send the prompt to llm
        response = query_rag(prompt)
        # show the llm response
        st.chat_message('assistant').markdown(response)
        # store the llm response in state
        st.session_state.messages.append(
            {'role':'assistant', 'content':response}
        )





if __name__ == "__main__":
    main()
