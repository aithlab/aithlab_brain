import streamlit as st
from datetime import datetime
from langchain_community.utilities import pebblo
from utils_gs import get_df_from_url, authorize_with_credentials, open_gs_by_url, save_df_to_gd, get_session_id, get_document
from utils_llm import get_retriever, get_llm_model, get_chat_chain

ip = pebblo.get_ip() 

# Streamlit
st.set_page_config(
    page_title="Taehwan's secretary",
    layout='centered', #wide
    page_icon=":robot_face:")

if 'store' not in st.session_state:
    store = {} # 세션 기록을 저장할 딕셔너리
    st.session_state.store = store
    
if 'df_qna' not in st.session_state:
    st.session_state.df_qna = None

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "language" not in st.session_state:
    st.session_state.language = None

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

llm_model = get_llm_model('gpt-4o')
document_id = st.secrets['document_id']
path = get_document(credentials, document_id)
retriever = get_retriever(path)

gc = authorize_with_credentials(credentials)
worksheet = open_gs_by_url(gc, url_gs, 'Sheet1')

if  st.session_state.language is None:
    st.title("Please select your language :smile:")
    language = st.selectbox(
        "Please select your language :smile:",
        ("한국어", "English"),
        placeholder='언어를 선택해주세요.',
        index=None,
        label_visibility='hidden',
        )

    if language is not None:
        st.session_state.language = language if language == 'English' else 'Korean'
        
        st.session_state.conversation = get_chat_chain(
            store=st.session_state.store, 
            session_id=st.session_state.session_id,
            retriever=retriever, 
            llm_model=llm_model,
            language='영어' if language == 'English' else '한글'
        )
        
        st.rerun()

else:
    st.title(
        # ":blue[T]aehwan's :blue[A]dvanced :blue[L]earning :blue[I]ntelligent :blue[A]ssistant (TALIA):robot_face:"
        ":blue[TALIA]:robot_face:"
        # "Ask me:robot_face: anything about :blue[Taehwan Kim]:male-technologist:" if st.session_state.language == 'English' else \
        # ":blue[김태환]:male-technologist:님에 대해 궁금한 내용을 저에게 물어봐주세요:robot_face:"
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi:raised_hand_with_fingers_splayed:, I'm Dr. Kim's secretary, TALIA. Please ask me anything about Dr. Kim!" if st.session_state.language == 'English' else \
            "안녕하세요:raised_hand_with_fingers_splayed: 저는 :blue[김태환]님의 비서인 TALIA입니다. 김태환님에 대해 궁금한 내용을 저에게 물어봐주세요!"
        }]
        
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
            _language = st.session_state.language
            with st.spinner("Thinking... :thinking_face:"):
                answer = _chain.stream(
                    {'question':query, 'language':_language},
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