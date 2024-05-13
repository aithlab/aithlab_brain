import streamlit as st
from datetime import datetime
from langchain_community.utilities import pebblo
from utils_gs import get_df_from_url, authorize_with_credentials, open_gs_by_url, save_df_to_gd, get_session_id, get_document
from utils_llm import get_retriever, get_llm_model, get_chat_chain

ip = pebblo.get_ip() 

# Streamlit
st.set_page_config(
    page_title="Taehwan's secretary",
    layout='wide', #centered
    page_icon=":robot_face:")

if 'store' not in st.session_state:
    store = {} # 세션 기록을 저장할 딕셔너리
    st.session_state.store = store
    
if 'df_qna' not in st.session_state:
    st.session_state.df_qna = None

if "conversation" not in st.session_state:
    st.session_state.conversation = None

credentials = {
    "type": st.secrets['type'],
    "project_id": st.secrets['project_id'],
    "private_key_id": st.secrets['private_key_id'],
    "private_key": st.secrets['private_key'],
    "client_email": st.secrets['client_email'],
    "client_id": st.secrets['client_id'],
    "auth_uri": st.secrets['auth_uri'],
    "token_uri": st.secrets['token_uri'],
    "auth_provider_x509_cert_url": st.secrets['auth_provider_x509_cert_url'],
    "client_x509_cert_url": st.secrets['client_x509_cert_url'],
    "universe_domain": st.secrets['universe_domain'],
    }

url_gs = st.secrets['url_gs']
_df_qna = get_df_from_url(url_gs)
st.session_state.df_qna = _df_qna

if 'session_id' not in st.session_state:
    st.session_state.session_id = get_session_id(_df_qna)

llm_model = get_llm_model()
document_id = st.secrets['document_id']
path = get_document(credentials, document_id)
retriever = get_retriever(path)
st.session_state.conversation = get_chat_chain(st.session_state.store, st.session_state.session_id,
                                          retriever, llm_model)

gc = authorize_with_credentials(credentials)
worksheet = open_gs_by_url(gc, url_gs, 'Sheet1')

st.title("Ask me:robot_face: anything about :blue[Taehwan Kim]:male-technologist:")

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "안녕하세요:raised_hand_with_fingers_splayed: 저는 :blue[김태환]님의 비서입니다. 김태환님에 대해 궁금한 내용을 저에게 물어봐주세요!"}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Please enter here")
if query:
    _session_id = st.session_state.session_id
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        chat_container = st.empty()
        _chain = st.session_state.conversation
        with st.spinner("Thinking..."):
            answer = _chain.stream(
                {'question':query},
                config={'configurable':{"session_id":_session_id}}
            )
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
    
    st.session_state.messages.append({"role": "assistant", "content": "".join(chunks)})
    
    _df_qna = st.session_state.df_qna
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _df_qna.loc[len(_df_qna)] = [ip, date, _session_id, query, "".join(chunks)]
    save_df_to_gd(_df_qna, worksheet)