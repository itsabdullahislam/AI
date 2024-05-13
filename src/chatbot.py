from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from src.prompt import PROMPT, CHAIN_PROMPT
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from src.utils import get_vector_store, stringify_searched_docs, return_splitter, format_docs
from langchain_community.document_transformers import LongContextReorder
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


class Chatbot:
    def __init__(self):
        self.prompt = PromptTemplate.from_template(template=PROMPT)
        model_kwargs = {'device': 'cpu'}
        self.embeddings = HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1",
                                                cache_folder='./model/embedding_model',
                                                model_kwargs=model_kwargs)
        self.retreiver = get_vector_store("documents").as_retriever(search_type='mmr')
        # embedding_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.70)
        # redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)                   

        max_tokens = 4096
        self.splitter = return_splitter(max_tokens, PROMPT)
        # pipeline_compressor = DocumentCompressorPipeline(
        #     transformers=[self.splitter, redundant_filter, embedding_filter]
        # )
        # self.compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=pipeline_compressor, base_retriever=self.retreiver
        # )
        self.reorder = LongContextReorder()
        self.llm = ChatOllama(
            model="llama2",
            temperature=0.0,
            num_ctx=max_tokens,
            # top_p=0.9,
            # top_k=4,
            num_gpu=4,
            repeat_penalty=1.0,
            # callback_manager=callback_manager,
            # verbose=True,
        )

        self.retreiver1 = get_vector_store("documents").as_retriever()
        # self.memory = ConversationSummaryMemory(llm=self.llm, return_messages=False, input_key='input',
        #                                         memory_key='chat_history')
        message_history = ChatMessageHistory()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )
        self.lcel = ({"question": itemgetter("question"),
                      "context": itemgetter('context'),
                      "chat_history": RunnableLambda(self.memory.load_memory_variables),
                      }
                     | self.prompt
                     | self.llm
                     | StrOutputParser()
                     )
        self.lcel1 = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | self.prompt
                | self.llm
                | StrOutputParser()
        )
        self.rag_chain = RunnableParallel(
            {"context": self.retreiver, "question": RunnablePassthrough(),
             "chat_history": RunnableLambda(self.memory.load_memory_variables)}
        ).assign(answer=self.lcel1)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="refine",
            retriever=self.retreiver1,
            memory=self.memory,
            condense_question_prompt=PromptTemplate.from_template(CHAIN_PROMPT),
            return_source_documents=True,
            verbose=True,
        )

    def chat(self, question: str) -> str:
        answer = self.lcel.invoke({
            "question": question,
            "context": self.get_ordered_docs(input_query=question),

        })
        self.memory.save_context({'input': question}, {'answer': answer})

        return answer

    def get_ordered_docs(self, input_query: str) -> str:
        docs = self.retreiver.get_relevant_documents(query=input_query)
        docs = self.splitter.transform_documents(docs)
        docs = self.reorder.transform_documents(docs)
        print(docs)
        doc_str = stringify_searched_docs(docs=docs)
        # print(doc_str)
        return docs
