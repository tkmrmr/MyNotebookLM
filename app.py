import gradio as gr
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.indexes import VectorstoreIndexCreator


def answer(file, text):
    loader = PDFPlumberLoader(file)

    embeddings = HuggingFaceEmbeddings(model_name="pkshatech/GLuCoSE-base-ja")
    llm = OllamaLLM(model="gemma2:27b-instruct-q4_K_M")

    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma, embedding=embeddings
    ).from_loaders([loader])

    query = text

    res = index.query(query, llm=llm)
    return res


demo = gr.Interface(fn=answer, inputs=["file", "text"], outputs="text")

demo.launch(share=True)
