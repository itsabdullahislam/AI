from src.prompt import PROMPT
from src.utils import return_splitter, get_vector_store
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader,UnstructuredPDFLoader
from fastapi import UploadFile
import os
from tempfile import NamedTemporaryFile
import PyPDF2

splitter = return_splitter(4096, PROMPT)


def transform(documents: [Document]):
    texts = splitter.split_documents(documents)
    print(texts)
    return texts


def read_pdf(file: UploadFile):
    vector_store = get_vector_store('documents')
    bytes_data = file.file.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        data = PyPDFLoader(tmp.name).load()
    os.remove(tmp.name)
    pages = data
    texts = transform(pages)
    vector_store.add_documents(texts, metadata={"source": file.filename})

    vector_store.persist()


def read_pdf_chain(file, filename):
    vector_store = get_vector_store("documents")
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ''
    for page in pdf.pages:
        pdf_text += page.extract_text()

    texts = transform([Document(pdf_text, metadata={"source": f"{filename}"})])
    vector_store.add_documents(texts)

    vector_store.persist()
