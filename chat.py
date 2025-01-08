from typing import Tuple

import gradio as gr
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

embeddings = HuggingFaceEmbeddings(model_name="pkshatech/GLuCoSE-base-ja")
llm = OllamaLLM(model="gemma2:27b-instruct-q4_K_M")


def process_pdf(file: str) -> Tuple[str, VectorStoreIndexWrapper | None]:
    try:
        loader = PDFPlumberLoader(file)

        index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma, embedding=embeddings
        ).from_loaders([loader])

        return "PDFの処理が完了しました．", index
    except Exception as e:
        return f"PDFの処理中にエラーが発生しました：{e}", None


def answer(message: str, history: list[str], index: VectorStoreIndexWrapper) -> str:
    if index is None:
        return "PDFがアップロードされていません"
    system_prompt = "回答はすべて日本語で行ってください。"
    res = index.query(system_prompt + message, llm=llm)
    return res


with gr.Blocks() as app:
    index = gr.State()

    gr.Markdown("# MyNotebookLM")
    gr.Markdown("アップロードしたPDFを参照してLLMが回答してくれるアプリ")
    gr.Markdown("PDFの処理を行ってからチャットに移ってください．")

    with gr.Tab("PDF処理"):
        with gr.Row():
            with gr.Column():
                file = gr.File(label="PDFをアップロード")
                upload_button = gr.Button("PDFを処理")
            result = gr.Textbox(label="処理結果")

    with gr.Tab("チャット"):
        gr.ChatInterface(
            fn=answer,
            type="messages",
            additional_inputs=[index],
            chatbot=gr.Chatbot(height=600, type="messages"),
        )

    upload_button.click(fn=process_pdf, inputs=[file], outputs=[result, index])

if __name__ == "__main__":
    app.launch(share=True)
