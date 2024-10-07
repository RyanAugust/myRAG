import os
from dotenv import load_dotenv
import logging
import torch
import langchain_community
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from langchain import PromptTemplate, HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
# from langchain_core.prompts import  PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, TextStreamer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

load_dotenv()

HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
print("USING HF TOKEN: %s" % HUGGING_FACE_TOKEN)

PDF_PATH = "./pdfs/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# model_name = "meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Llama-3.2-1B"

DEFAULT_SYSTEM_PROMPT = """
Based on the information in this document provided in context, answer the question as accurately as possible in 1 or 2 lines. If the information is not in the context,
respond with "I don't know" or a similar acknowledgment that the answer is not available.
""".strip()
SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. Do not provide commentary or elaboration more than 1 or 2 lines."


# # Quanitisize your model dtype
# bnb_config = BitsAndBytesConfig(
    # load_in_4bit=False,
    # bnb_4bit_use_double_quant=True, 
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
# )
# # Set token using ENV variable
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     # token=HUGGING_FACE_TOKEN,
#     local_files_only=True
#     # quantization_config=bnb_config
# )
# def test_model():
#     prompt = f"What's the capital of the United States of America?"
#     inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
#     inputs = inputs.to('cpu')  # Ensure inputs are on CPU

#     output = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         num_beams=1,
#         do_sample=False,
#         temperature=1
#     )

#     answer = tokenizer.decode(output[0], skip_special_tokens=True)

#     print(answer)

def load_pdfs(path:os.PathLike):
    loader = PyPDFDirectoryLoader("./pdfs/")
    docs = loader.load()
    logger.info("found %s documents" % len(docs))
    return docs

def extract_text_from_pdfs(docs: list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    return texts

def make_embeddings(texts):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": DEVICE}
    )

    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
    return db

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
    [INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {prompt} [/INST]
    """.strip()

def preprocess_pdfs(path:os.PathLike):
    docs = load_pdfs(path=path)
    texts = extract_text_from_pdfs(docs=docs)
    db = make_embeddings(texts=texts)
    return db


def main():
    db = preprocess_pdfs(path=PDF_PATH)
   # # Quanitisize your model dtype
    # bnb_config = BitsAndBytesConfig(
        # load_in_4bit=False,
        # bnb_4bit_use_double_quant=True, 
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # # Set token using ENV variable
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HUGGING_FACE_TOKEN,
        # local_files_only=True
        # quantization_config=bnb_config
    )
    template = generate_prompt("""\n{context}\nQuestion: {question}""", system_prompt=SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    ask = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return ask


if __name__ == "__main__":
    ask = main()
    question = input("?: ")
    result = ask("Give me a TLDR of this document") # ask(question)
    for k,v in result.items():
        print(f"[{k}]\n\t{v}\n")
