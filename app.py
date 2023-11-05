from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.llms import CTransformers
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
import chainlit as ct
import asyncio

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DB_FAISS_PATH = 'faiss-meditations-index'

custom_prompt_template = """You are an AI assistant helping users to understand different meditation techniques. Your knowledge base is a book named Vigyan Bhairava Tantra. Keep your answers short and simple. If question is not clear ask user to rephrase the question. Generate sample questions for the user to ask.

Context: {context}
Question: {context}

Response for Questions asked.
answer:
"""

def create_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", 'question'])
    return prompt

#retreivel Chain
def get_response_from_qa_chain(lm, prompt, db):
    retreival_chain = RetrievalQA.from_chain_type(llm=lm,
                                                chain_type="stuff",
                                                retriever=db.as_retriever(search_kwargs={"k": 1}),
                                                return_source_documents=True,
                                                chain_type_kwargs={"prompt": prompt})
    return retreival_chain


#Loading the local model into LLM
def load_llama2_llm():
    # Load the model 1lama-2-7b-chat.ggmlv3.q8_0.bin that was downloaded locally
    llm = CTransformers(
        model = ".venv/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 32000,
        temperature = 0.5
    )

    # model_repo = 'daryl149/llama-2-7b-chat-hf'

    # tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_repo,
    #     load_in_4bit=True,
    #     device_map='auto',
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True
    # )

    # max_len = 2048
    # pipe = pipeline(
    #     task = "text-generation",
    #     model = model,
    #     tokenizer = tokenizer,
    #     pad_token_id = tokenizer.eos_token_id,
    #     max_length = max_len,
    #     temperature = 0,
    #     top_p = 0.95,
    #     repetition_penalty = 1.15
    # )

    # llm = HuggingFacePipeline(pipeline = pipe)
    
    return llm


# answering bot creation
def answering_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llama2_llm()
    message_prompt = create_prompt()
    response = get_response_from_qa_chain(llm, message_prompt, vectorstore)

    return response

#display the result of the question asked
def final_result(query):
    bot_result = answering_bot()
    bot_response = bot_result({'query': query})
    return bot_response

#chainlit code you can refer to the chainlit.io website for more details.
@ct.on_chat_start
async def start():
        chain = answering_bot()
        msg = ct.Message(content="The bot is getting initialized, please wait!!")
        await msg.send()
        msg.content = "Q&A bot is ready. Ask questions on the documents indexed?"
        await msg.update()
        ct.user_session.set("chain", chain)

@ct.on_message
async def main(message):
    chain = ct.user_session.get("chain")
    cb = ct.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    print(answer)
    await ct.Message(content=answer).send()