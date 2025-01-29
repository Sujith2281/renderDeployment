from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("API Key loaded successfully")
else:
    print("Error: API Key not found")



import streamlit as st
import tempfile
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader, PyPDFLoader, TextLoader, UnstructuredFileLoader,
    CSVLoader, Docx2txtLoader
)
from langchain_community.embeddings import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def load_files(uploaded_files):
    combined_documents = []
    if uploaded_files:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]  

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='_' + uploaded_file.name) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == 'txt':
                loader = TextLoader(temp_file_path)
            elif file_extension == 'csv':
                loader = CSVLoader(temp_file_path)
            elif file_extension == 'docx':
                loader = Docx2txtLoader(temp_file_path)
            else:
                loader = UnstructuredFileLoader(temp_file_path)

            documents = loader.load()
            combined_documents.extend(documents)

    return combined_documents

def load_urls(url_input):
    combined_documents = []
    if url_input:
        urls = [url.strip() for url in url_input.split(',')]

        for url in urls:
            loader = WebBaseLoader(url)
            documents = loader.load()
            combined_documents.extend(documents)
    return combined_documents

def process_document(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    doc_chunk = text_splitter.split_documents(doc)

    return doc_chunk

def create_vector_store(doc_chunk):
    embeddings = OpenAIEmbeddings()  
    database = FAISS.from_documents(doc_chunk, embeddings)
    return database.as_retriever()

if "history" not in st.session_state:
    st.session_state.history = []

chat_history = []
store = {}
chat_history = ChatMessageHistory()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# LLM Initialization
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Contextualize question 
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# Answer question 
system_prompt = (
    "you are assistant for conversational bot ."
    "Use the following pieces of retrieved context to answer "
    "the question. do not use own knowlege to answer ,If you don't know the answer, just say that you "
    "don't know. Use two sentences maximum and keep the "
    "answer concise. use bullet points when it is require"
    "if query include words like 'in detail','describe','explain', then give reponse accordingly"
    
    
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# Streamlit UI
st.title("RAG Chatbot")

uploaded_file = st.file_uploader("Upload a file (PDF/TXT/Other)", type=["pdf", "txt", "docx", "csv"], accept_multiple_files=True)
url_ip = st.text_input("Or enter a URL:")

if "conversational_rag_chain" not in st.session_state:
    st.session_state.conversational_rag_chain = None

# Function to load and process documents, create vector store
def initialize_retrieval_chain(uploaded_file, url_ip):
    if uploaded_file or url_ip:
        if uploaded_file:
            with st.spinner('Loading files...'):
                loads = load_files(uploaded_files=uploaded_file)
        elif url_ip:
            with st.spinner('Loading URLs...'):
                loads = load_urls(url_input=url_ip)

        doc = loads

        if doc:
            with st.spinner('Processing document...'):
                doc_chunk = process_document(doc)

            with st.spinner('Creating vector store...'):
                retriever = create_vector_store(doc_chunk)

            history_aware_retriever = create_history_aware_retriever(
                                llm, retriever, contextualize_q_prompt)

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # Statefully manage chat history
            st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                retrieval_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

if st.session_state.conversational_rag_chain is None:
    initialize_retrieval_chain(uploaded_file, url_ip)

question = st.text_input("Ask a question about the document:")

print(st.session_state.history)

if question and st.session_state.conversational_rag_chain:
    with st.spinner('Generating response...'):
        response = st.session_state.conversational_rag_chain.invoke({"input": question}, config={"configurable": {"session_id": "abc123"}})
        answer = response['answer']
        st.session_state.history.append({"question": question, "answer": answer})

if st.session_state.history:
    chat_history = ''
    for idx, chat in enumerate(st.session_state.history):
        chat_history += f"**Question {idx + 1}**: {chat['question']}\n\n"
        chat_history += f"**Response {idx + 1}**: {chat['answer']}\n\n"

    # Create the chat history text area
    chat_history_text_area = st.text_area("Chat History1", value=chat_history, height=400, max_chars=None, key="chat_history_text_area")

    # Scroll to the bottom of the chat history
    st.markdown(
        """
        <script>
        window.onload = function() {
            var textarea = window.parent.document.querySelector('textarea[aria-label="Chat History"]');
            if (textarea) {
                textarea.scrollTop = texta
        };
        </script>
        """,
        unsafe_allow_html=True,
    )

# Trigger scroll to bottom whenever a new message is added
st.session_state.history_updated = st.session_state.get('history_updated', False)

if st.session_state.history_updated:
    st.session_state.history_updated = False
    st.experimental_rerun()

def add_message(question, answer):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'question': question, 'answer': answer})
    st.session_state.history_updated = True
