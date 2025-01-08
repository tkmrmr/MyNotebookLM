import gradio as gr
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.indexes import VectorstoreIndexCreator

embeddings = HuggingFaceEmbeddings(model_name="pkshatech/GLuCoSE-base-ja")
llm = OllamaLLM(model="gemma2:27b-instruct-q4_K_M")


def answer(file: gr.File, query: str) -> str:
    loader = PDFPlumberLoader(file)

    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma, embedding=embeddings
    ).from_loaders([loader])

    res = index.query(query, llm=llm)
    return res


with gr.Blocks() as app:
    gr.Markdown("# MyNotebookLM")
    gr.Markdown("アップロードしたPDFを参照してLLMが回答してくれるアプリ")

    with gr.Row():
        with gr.Column():
            pdf_file = gr.File(label="PDFをアップロード")
            query = gr.Textbox(label="質問")
            button = gr.Button("送信")

        output = gr.Textbox(label="回答")

    button.click(fn=answer, inputs=[pdf_file, query], outputs=[output])

if __name__ == "__main__":
    app.launch(share=True)
