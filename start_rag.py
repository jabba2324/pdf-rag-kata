import gradio as gr
from rag import basic_rag_pipeline

# Launch Gradio interface
gr.Interface(
    fn=basic_rag_pipeline,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the Universal planning documents..."),
    outputs=gr.Textbox(lines=20),
    title="Universal Planning Documents RAG Chatbot",
    description="Ask questions about Universal's Entertainment Resort Complex planning application in Bedford."
).launch()