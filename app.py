import streamlit as st
from sklearnrag.generate import QueryAgent
from langchain.memory import ConversationBufferMemory
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title='Sklearn QA Bot', page_icon='📋', layout="wide")

icon_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/2560px-Scikit_learn_logo_small.svg.png"
st.markdown(f"""
    <h1 style="text-align: center;">
        <img src="{icon_url}" alt="Icon" style="vertical-align: middle; height: 112px; margin-right: 50px;">
        <span style="color: #F7931E; font-family: 'Sans Serif';">{"Scikit-Learn QA Bot"}</span>
    </h1>
""", unsafe_allow_html=True)
st.write("\n")


system_content = "Answer the query purely using the context provided. First look at the API Reference not the examples. Do not try to make things up. Be succinct."
agent = QueryAgent(
    embedding_model_name="thenlper/gte-large",
    llm="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_context_length=32768,
    system_content=system_content)

if "messages" not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def stream_responses():
    result = agent(query=st.session_state['messages'][-1]['content'], stream=True, num_chunks=7)
    for content in result['answer']:
        yield content

    yield "\n\nRelated Sources:\n"

    for i, source in enumerate(result['sources']):
        yield f"{i+1}. {source}\n"


prompt = st.chat_input("Hi, I'm your AI assistant to help you in answering Scikit-Learn related queries. Ask me anything!")

if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message = st.write_stream(stream_responses())
        st.session_state['messages'].append({"role": "assistant", "content": message})









