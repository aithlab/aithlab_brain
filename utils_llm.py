import streamlit as st
from langchain.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

@st.cache_resource
def get_llm_model(model_name="gpt-3.5-turbo"):
    model = ChatOpenAI(
        temperature=0.0,
        max_tokens=None, #채팅 완성에서 생성할 토큰의 최대 개수
        model_name=model_name,
        streaming=True,
        api_key=st.secrets['openai_api_key']
    )
    return model

def get_vectorstore(loader, vecstore_name):
    cache_dir = LocalFileStore("./.cache/")
    
    if vecstore_name == 'Chroma':
        embeddings = OpenAIEmbeddings(api_key=st.secrets['openai_api_key'])
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir) 
        data = loader.load()
        vectorstore = Chroma.from_documents(data, cached_embeddings)
        
    elif vecstore_name == 'FAISS':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100
        )
        pages = loader.load_and_split()
        texts = text_splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )  
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(texts, cached_embeddings)
        
    return vectorstore

@st.cache_resource
def get_retriever(path):
    loader = TextLoader(path)
    
    vectorstore = get_vectorstore(loader, 'Chroma')
    retriever = vectorstore.as_retriever()
    return retriever

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 김태환 박사님의 비서입니다. 김태환 박사님에 대한 질문을 대신 답해주는 역할입니다.(답변은 한글로해주세요.) 질문을 요약해주세요.
            """,
        ),
        ("human", "{question}"),
    ]
)

map_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 김태환 박사님의 비서입니다. 김태환 박사님에 대한 질문을 대신 답해주는 역할입니다.(답변은 한글로해주세요.) 질문에 답하기 위해 필요한 내용을 다음 문장에서 찾아서 내용을 정리해주세요(만약 관련된 내용이 없다면, 아무것도 반환하지 마세요.):
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

reduce_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
            당신은 김태환 박사님의 비서입니다. 김태환 박사님에 대한 질문을 대신 답해주는 역할입니다.(답변은 한글로해주세요.) 다음에 주어진 문장들을 이용해 답변을 작성해주세요(1.논문의 경우에는 한글로 번역하지 말아주세요. 그리고 저널명까지 함께 보여주세요. 2.질문에 답변하기 위한 정보가 충분하지 않은 경우에는 김태환 박사에게 연락할 수 있도록 안내해주세요.):
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

refine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            당신은 김태환 박사님의 비서입니다. 김태환 박사님에 대한 질문을 대신 답해주는 역할입니다.(답변은 한글로해주세요.) 다음의 내용을 공손한 말투로 다시 작성해주세요(작성한 내용만 반환해주세요):
            {context}
            """
        )
    ]
)

class mapping:
    def __init__(self, retriever, map_chain):
        self._retriever = retriever
        self._map_chain = map_chain
    
    def map_docs(self, inputs):
        question = inputs["summary"].content
        documents = self._retriever.invoke(question)
        return "\n\n".join(
            self._map_chain.invoke({"context": doc.page_content, "question": question}).content
            for doc in documents
        )

class session_history:
    def __init__(self, store, session_id):
        self._store = store
        self._session_id = session_id

    # 세션 ID를 기반으로 세션 기록을 가져오는 함수
    def get(self, _):
        if self._session_id not in self._store: # 세션 ID가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            self._store[self._session_id] = ChatMessageHistory()
        
        return self._store[self._session_id]

# @st.cache_resource
def get_chain(retriever, llm_model):
    summary_chain = summary_prompt | llm_model
    map_chain = map_prompt | llm_model
    _mapping = mapping(retriever, map_chain)
    map_results = {"summary": summary_chain} | RunnableLambda(_mapping.map_docs)
    reduce_chain = {"context": map_results, "question": RunnablePassthrough()} | reduce_prompt | llm_model
    refine_chain = {"context": reduce_chain} | refine_prompt | llm_model | StrOutputParser()
    return refine_chain

def get_chat_chain(store, session_id, retriever, llm_model):
    refine_chain = get_chain(retriever, llm_model)
    history = session_history(store, session_id)
    
    with_message_history = (
        RunnableWithMessageHistory(
            refine_chain, # 실행할 runnable 객체
            history.get, # 세션 기록을 가져오는 함수
            input_messages_key='question', #입력 메시지의 키
            history_messages_key='history' # 기록 메시지의 키
        )
    )
    return with_message_history