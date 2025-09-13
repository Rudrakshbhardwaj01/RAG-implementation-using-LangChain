from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Load PDF
loader = PyPDFLoader("YOUR-PDF'S-PATH")
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
split_docs = text_splitter.split_documents(docs)

# Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="DB-NAME")
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Summarize the following context into 3–5 sentences. 
Be clear and concise, and only use the information from the context. 
Do not add anything that is not present.

Context:
{context}

Summary:
"""
)

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",   # or "google/flan-ul2" if your GPU can handle it
    task="text2text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=300,
        do_sample=False,   # deterministic summarization
    )
)


# No ChatHuggingFace, just use llm directly
user_question = input("Ask me anything about your PDF: ")

# Retrieve relevant docs
results = vectorstore.similarity_search(user_question, k=1)
context = "\n\n".join([doc.page_content for doc in results])

# Format final prompt
final_prompt = prompt_template.format(context=context)

# Get answer
answer = llm.invoke(final_prompt)

print("\nAnswer:")
print(answer)

