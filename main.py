import streamlit as st
import re
import tiktoken
from transformers import pipeline
from TOONIE import models

from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.schema import Document

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain_community.llms import HuggingFacePipeline


model_name = "test"

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")
    if process:
        vetorestore = get_vectorstore(uploaded_files)

        st.session_state.conversation = get_conversation_chain(vetorestore, model_name)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(result['context'])
                    # st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    # st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)

                # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def extract_patterns(full_text, pattern):
    matches = re.finditer(pattern, full_text)
    start_positions = [match.start() for match in matches]
    return start_positions

def process_text_lump(file_name, title, text_lump):
    rule_text_list = []

    # "부칙" 분리
    separate_match = re.search(r'부\s*\S*\s*칙', text_lump)
    separate_text = text_lump[separate_match.start():] if separate_match else ""
    text_lump = text_lump[:separate_match.start()] if separate_match else text_lump

    # rule 분리 ex) "제 1조"
    rule_start_positions = extract_patterns(text_lump, r'제\s*\d+\s*조\s*[(【{[]')

    for i, start_index in enumerate(rule_start_positions):
        end_index = rule_start_positions[i + 1] if i < len(rule_start_positions) - 1 else None
        rule_text = text_lump[start_index:end_index]

        final_rule_text = Document(page_content=file_name + "\n" + title + "\n" + rule_text)
        rule_text_list.append(final_rule_text)

    rule_text_list.append(Document(page_content=file_name + "\n" + separate_text))

    return rule_text_list


def to_text_list(file_name, documents):
    full_text = ""

    for page in documents:
        text = page.page_content
        full_text += text

    # 제목 추출 ex) "제 1장 총칙"
    title_positions = extract_patterns(full_text, r'제 \d+ 장 [^\n]+')

    final_text_list = []
    if not title_positions:
        final_text_list = process_text_lump(file_name, "", full_text)

    else:
        for i, title_index in enumerate(title_positions):
            title = re.search(r'제 \d+ 장 [^\n]+', full_text[title_index:]).group()

            if i == len(title_positions) - 1:
                text_lump = full_text[title_index:]
            else:
                end_index = title_positions[i + 1]
                text_lump = full_text[title_index:end_index]

            rule_text_list = process_text_lump(file_name, title, text_lump)
            final_text_list += rule_text_list

    return final_text_list


def get_vectorstore(docs):
    texts = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load()
        final_text_list = to_text_list(file_name, documents)
        texts += final_text_list

    # 한국어 임베딩 모델
    model_name = "jhgan/ko-sbert-nli"
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    # FAISS를 사용한 검색 인덱스 구축
    # 해당 DB를 검색 하는데 사용할 것 이라는 선언
    vectordb = FAISS.from_documents(texts, hf)

    return vectordb


def get_conversation_chain(vetorestore, model_name):
    tokenizer, model, prompt_template = models.get_models(model_name)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,  # 0으로 되어 있으면 일관성 있는 대답, 1로 되어 있으면 다양한 대답
        return_full_text=True,  # 전체 텍스트 출력
        max_new_tokens=500,
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        # retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
        retriever=vetorestore.as_retriever(search_type="similarity_score_threshold",
                                           search_kwargs={"score_threshold": 0.4}),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain


if __name__ == '__main__':
    main()