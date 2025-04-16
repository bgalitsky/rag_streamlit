import os
import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import GigaChat
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GIGACHAT_API_KEY")

# Sample questions (edit to match your documents!)
SAMPLE_QUESTIONS = [
    "Какие температурные параметры может регулировать пушка в автоматическом режиме?",
    "Ставрополье: заморозки угрожают потерей до 50% урожая чего?",
    "В какой период были заморозки?",
    "Тепловые пушки в парниках - основные затраты",
    "На сколько градусов ночью снижается температура в парниках?",
    "В оранжереях направленный воздушный поток что делает?",
    "Куда воздушная струя должна быть направлена? "
]

@st.cache_resource
# Шаг 1: Загрузка текста из файлов в директории
def load_documents_from_directory(directory):
    """
    Загружает текст из всех файлов в указанной директории.
    Поддерживаются форматы: .txt, .pdf, .docx.

    :param directory: путь к директории с документами
    :return: список текстовых документов
    """
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Обработка текстовых файлов (.txt)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())

        # Обработка PDF файлов (.pdf)
        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)

        # Обработка DOCX файлов (.docx)
        elif filename.endswith(".docx"):
            doc = DocxDocument(filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            documents.append(text)

    return documents


# Шаг 2: Создание векторного представления документов
def create_vector_store(documents):
    """
    Создает векторный индекс документов с помощью FAISS.

    :param documents: список текстовых документов
    :return: объект FAISS для поиска
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store


# 4. Setup RAG pipeline
def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    giga = GigaChat(
        # Для авторизации запросов используйте ключ, полученный в проекте GigaChat API
        credentials=api_key,
        verify_ssl_certs=False,
    )
    qa_chain = RetrievalQA.from_chain_type(llm=giga, retriever=retriever, return_source_documents=True)
    return qa_chain

def ask_question(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"]

# Streamlit UI
st.set_page_config(page_title="Заморозки в Ставропольском крае", layout="centered")
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f7f7f7;
        border-radius: 1rem;
        margin-bottom: 1rem;
    }
    .chat-bubble {
        max-width: 80%;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 1rem;
        font-size: 1rem;
        line-height: 1.4;
    }
    .user {
        background-color: #dcf8c6;
        align-self: flex-end;
        margin-left: auto;
    }
    .bot {
        background-color: #ffffff;
        border: 1px solid #e1e1e1;
        align-self: flex-start;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🤖Заморозки в Ставропольском крае")
# File path setup (edit your folder path here)
DOCS_PATH = "resources"

with st.spinner("Loading documents and building vector store..."):
    docs = load_documents_from_directory(DOCS_PATH)
    vector_store = create_vector_store(docs)
    qa_chain = create_qa_chain(vector_store)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#user_input = st.text_input("Ваш вопрос", placeholder="Ask me anything from the docs...")

# Sample Questions UI
st.markdown("### 💡 Примеры вопросов")
cols = st.columns(2)
for i, question in enumerate(SAMPLE_QUESTIONS):
    if cols[i % 2].button(question):
        st.session_state.selected_question = question

# Chat input field
user_input = st.chat_input("Введите ваш вопрос здесь...")

# Use selected sample if available
if "selected_question" in st.session_state:
    user_input = st.session_state.selected_question
    del st.session_state.selected_question

if user_input:
    with st.spinner("Searching and generating answer..."):
        answer = ask_question(qa_chain, user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, message in st.session_state.chat_history:
    css_class = "user" if role == "user" else "bot"
    st.markdown(f'<div class="chat-bubble {css_class}">{message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)