from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate


loader = PyPDFLoader("YOUR-PDF'S-PATH")
docs=loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      
    chunk_overlap=100,    
)
split_docs = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="DB-NAME")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert AI assistant specialized in understanding and explaining technical material. 
Use the information in the context below to answer the question as accurately and clearly as possible.

Guidelines:
1. Only use information explicitly present in the context. Do NOT hallucinate or assume facts not in the text.
2. Summarize and simplify complex concepts for easy understanding, but keep technical accuracy.
3. If the answer is not contained in the context, respond with: "The answer is not present in the provided context."
4. Provide step-by-step explanations if the question involves a process or procedure.
5. Keep the answer concise, but complete enough to cover the question.

Context: {context}

Question: {question}

Answer (only provide the final answer, do not repeat the question or context):
"""
)

from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from transformers import pipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=300,
    )
)

model = ChatHuggingFace(llm=llm)

user_question = input("Ask me anything about your PDF: ")

results = vectorstore.similarity_search(user_question, k=4)
context = "\n\n".join([doc.page_content for doc in results])

final_prompt = prompt_template.format(context=context, question=user_question)

answer = model.invoke(final_prompt)

print("\nAnswer:")
print(answer.content)

