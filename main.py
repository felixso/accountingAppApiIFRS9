__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

# Groq API-Schlüssel aus secrets.toml laden
groq_api_key = st.secrets["GROQ_API_KEY"]

# Groq API-Schlüssel setzen
import os
os.environ["GROQ_API_KEY"] = groq_api_key

# Dokumente laden und verarbeiten
#loader = DirectoryLoader("path/to/your/documents", glob="**/*.txt")
loader = DirectoryLoader(
    path="./IFRS_TEXT",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()
print(f"Anzahl der geladenen Dokumente: {len(documents)}")
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embeddings erstellen und Vektorstore initialisieren
embeddings = HuggingFaceEmbeddings()
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#vectorstore = Chroma.from_documents(texts, embeddings)
#vectorstore = Chroma.from_documents(documents=texts, embedding=HuggingFaceEmbeddings())
#vectorstore = Chroma.from_documents(documents=texts, embeddings,persist_directory="vector_db_dir")
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)


# Groq LLM initialisieren
#llm = Groq(model_name="llama2-70b-4096")
#llm = ChatGroq(model_name="llama2-70b-4096")
llm = ChatGroq(model="llama-3.1-70b-versatile",temperature=0)


# Konversationsgedächtnis erstellen
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = ConversationBufferMemory(
    llm=llm,
    output_key="answer",
    memory_key="chat_history",
    return_messages=True
)


# RAG-Chain erstellen
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type = "stuff",
    memory=memory,
    verbose = True,
    return_source_documents=True
)

# Streamlit UI

st.set_page_config(
    page_title="Chatbot",
    layout="centered"
)
st.title("RAG Chatbot mit Groq und LangChain")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Was möchten Sie wissen?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = qa({"question": prompt})
        st.markdown(response['answer'])
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
