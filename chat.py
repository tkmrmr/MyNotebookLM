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


def generate_summary(index: VectorStoreIndexWrapper) -> str:
    if index is None:
        return "PDFがアップロードされていません"
    query = "この文書の概要を日本語で答えてください。"
    summary = index.query(query, llm=llm)
    return summary


def answer(message: str, history: list[str], index: VectorStoreIndexWrapper) -> str:
    if index is None:
        return "PDFがアップロードされていません"
    query = f"""この文書の内容を基に、質問に対する回答を日本語で生成してください。回答は文書で明示されている情報のみに基づき、文書の範囲外の情報や推測を含めないでください。必要に応じて、文書内から具体的な引用を含めても構いません。

    質問：
    {message}

    注意事項：
    ・回答はすべて日本語で生成してください。
    ・文書に記載されていない情報については、「文書内にその情報は記載されていません」と回答してください。
    ・回答を生成する際は、文書内の情報に忠実であることを優先してください。

    回答：
    """
    answer = index.query(query, llm=llm)
    return answer


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
                generate_summary_button = gr.Button("概要を表示")
            with gr.Column():
                result = gr.Textbox(label="処理結果")
                summary = gr.Textbox(label="概要")

    with gr.Tab("チャット"):
        gr.ChatInterface(
            fn=answer,
            type="messages",
            additional_inputs=[index],
            chatbot=gr.Chatbot(height=600, type="messages"),
        )

    upload_button.click(fn=process_pdf, inputs=[file], outputs=[result, index])
    generate_summary_button.click(
        fn=generate_summary, inputs=[index], outputs=[summary]
    )

if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0")
