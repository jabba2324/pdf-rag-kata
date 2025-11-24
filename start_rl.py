import gradio as gr
from rl_vs_rag import compare_rag_approaches
# Launch Gradio interface
gr.Interface(
    fn=compare_rag_approaches,
    inputs=[
        gr.Textbox(label="query_text", lines=2, placeholder="Enter your query here"),
        gr.Textbox(label="ground_truth", lines=2, placeholder="Enter the correct answer here")
        ],
    outputs=gr.Textbox(lines=20),
    title="Universal Planning Documents RAG Chatbot",
    description="Ask questions about Universal's Entertainment Resort Complex planning application in Bedford."
).launch()
