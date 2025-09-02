import os
import google as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

print("Started Execution.")
# ✅ Configure Gemini
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["GOOGLE_API_KEY"] = "AIzaSyB82bt4BVcALdb4uz0eDcbykdScvxe4dUY"

# Step 1: Load PDF
loader = PyPDFLoader("IndianConstitution.pdf")
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings using Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Step 4: Store in vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 5: Setup Retriever + LLM
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0),
    retriever=retriever
)

# Step 6: Ask questions
query = "Summarize this PDF in 5 bullet points."
answer = qa.invoke(query)

print("Answer:", answer)
