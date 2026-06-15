import gradio as gr
from search import search_patents, get_novelty_label

def check_novelty(user_idea):
    if not user_idea.strip():
        return "Please enter an idea.", ""

    results, novelty = search_patents(user_idea, top_k=3)
    label = get_novelty_label(novelty)

    # Top 3 patents output
    output = "### Top Similar Patents\n\n"
    for r in results:
        bar = "" * int(r["similarity"] // 5)
        output += f"**#{r['rank']}. {r['title']}**\n"
        output += f"Patent ID: `{r['patent_id']}`\n"
        output += f"Similarity: `{r['similarity']}%`  {bar}\n\n"
        output += "---\n"

    # Novelty score output
    icon = "" if label["color"] == "green" else "" if label["color"] == "orange" else "❌"
    novelty_out  = f"## {icon} Novelty Score: {novelty}%\n\n"
    novelty_out += f"### {label['label']}\n"
    novelty_out += f"{label['message']}\n\n"
    novelty_out += f"**Formula:** 100% − {results[0]['similarity']}% (highest match) = **{novelty}%**"

    return output, novelty_out


# Example ideas
examples = [
    ["A system that uses AI sensors to automatically adjust traffic signal timings based on real-time vehicle density"],
    ["A wearable device that monitors blood glucose levels using non-invasive infrared sensors"],
    ["A drone that delivers medicine to remote areas using GPS navigation"],
    ["A blockchain-based system for securing medical records and patient data"],
    ["A solar panel that automatically adjusts its angle to maximize energy absorption"],
]

# Gradio UI
with gr.Blocks(title="Patent Novelty Checker", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # Patent Novelty Checker
    ### Deep Learning-Based Patent Similarity & Novelty Detection
    Powered by **Sentence-BERT (MiniLM-L6-v2)** + **FAISS** · Searching **3,488 real patents**
    """)

    with gr.Row():
        with gr.Column(scale=2):
            idea_input = gr.Textbox(
                lines       = 5,
                placeholder = "Describe your invention idea in detail...",
                label       = "Your Invention Idea"
            )
            with gr.Row():
                submit_btn = gr.Button(" Check Novelty", variant="primary", scale=2)
                clear_btn  = gr.Button(" Clear", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### ℹ How it works")
            gr.Markdown("""
            1. Your idea is encoded by **SBERT** into a 384-dim vector
            2. **FAISS** searches all 3,488 patent vectors
            3. Top 3 most similar patents are returned
            4. **Novelty = 100% − highest similarity**
            """)

    with gr.Row():
        patents_output = gr.Markdown(label="Top Similar Patents")
        novelty_output = gr.Markdown(label="Novelty Score")

    gr.Markdown("###  Try these examples:")
    gr.Examples(
        examples = examples,
        inputs   = idea_input,
        label    = "Click any example to load it"
    )

    submit_btn.click(
        fn      = check_novelty,
        inputs  = idea_input,
        outputs = [patents_output, novelty_output]
    )
    clear_btn.click(
        fn      = lambda: ("", "", ""),
        outputs = [idea_input, patents_output, novelty_output]
    )

if __name__ == "__main__":
    app.launch(share=False)