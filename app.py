from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
import chainlit as cl
from typing import List
from src.data_parse import read_pdf_chain
from src.chatbot import Chatbot


@cl.on_chat_start
async def on_chat_start():
    # Sending an image with the local file path
    
    files = None
    await cl.Message(content="Hello there!", elements=elements).send()
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    print(file)
    read_pdf_chain(file, file.name)
    msg = cl.Message(content=f"File processed `{file.name}`...")
    await msg.send()
    chat = Chatbot()

    cl.user_session.set("chain", chat.chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke({"question": message.content}, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    await cl.Message(content=answer, elements=text_elements).send()
