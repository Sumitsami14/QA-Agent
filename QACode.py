print("QA Agent started.")

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "AIzaSyB82bt4BVcALdb4uz0eDcbykdScvxe4dUY"


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
messages = [
    ("system", "I want to design an resume."),
    ("human", "Guide me"),
]
response = llm.invoke(messages)
print(response.content)
# print(response)



embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector = embeddings.embed_query("hello, world!")  # returns a vector


