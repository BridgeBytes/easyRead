"""
EasyRead Gradio Frontend

A web interface for the EasyRead document simplification service.
Converts complex text into Easy Read format with pictogram icons.
"""

import gradio as gr
import tempfile

from utils.backend import BackendClient, RevisedSentence
from utils.docx_export import export_to_docx_bytes
from utils.markdown_export import export_to_markdown_bytes


# Initialize backend client
client = BackendClient()

# Global state storage
state = {
    "title": "",
    "revised_sentences": [],
    "request_id": "",
    "icons": [],
    "image_data": {},
}


def simplify_text(text: str, context: str, unalterable_terms: str):
    """Send text to backend for simplification."""
    if not text.strip():
        return (
            gr.update(visible=True, value="Please enter some text to simplify."),
            gr.update(visible=False),  # review section
            gr.update(visible=False),  # results section
        )

    try:
        response = client.simplify_text(
            text=text,
            custom_context=context if context.strip() else None,
            unalterable_terms=unalterable_terms if unalterable_terms.strip() else None,
        )

        state["title"] = response.title
        state["revised_sentences"] = response.revised_sentences

        # Format sentences for display and editing
        df_data = []
        for idx, s in enumerate(response.revised_sentences):
            df_data.append([
                idx + 1,
                s.sentence,
                s.image_prompt,
                s.highlighted,
            ])

        validation_text = f"""**Missing Information:** {response.validation.get('missing_info', 'None')}

**Extra Information:** {response.validation.get('extra_info', 'None')}

**Other Feedback:** {response.validation.get('other_feedback', 'None')}"""

        return (
            gr.update(visible=False),  # error
            gr.update(visible=True),   # review section
            gr.update(visible=False),  # results section
            f"## {response.title}",
            validation_text,
            df_data,
        )

    except Exception as e:
        error_msg = str(e).lower()
        if any(x in error_msg for x in ["503", "overload", "unavailable", "timeout", "connection", "error"]):
            user_message = "The AI service is temporarily unavailable. Please try again."
        else:
            user_message = "Something went wrong. Please try again."

        return (
            gr.update(visible=True, value=user_message),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            "",
            None,
        )


def update_sentences_from_table(table_data):
    """Update internal state from edited table data."""
    if table_data is None:
        return

    state["revised_sentences"] = []

    # Handle different Gradio Dataframe formats
    rows = []
    if hasattr(table_data, "values"):
        rows = table_data.values.tolist()
    elif isinstance(table_data, dict) and "data" in table_data:
        rows = table_data["data"]
    elif isinstance(table_data, list):
        rows = table_data
    else:
        return

    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        state["revised_sentences"].append(
            RevisedSentence(
                sentence=str(row[1]),
                image_prompt=str(row[2]),
                highlighted=bool(row[3]),
            )
        )


def generate_icons(table_data):
    """Generate icons for approved sentences."""
    update_sentences_from_table(table_data)

    if not state["revised_sentences"]:
        return (
            gr.update(visible=True, value="No sentences to generate icons for."),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
        )

    try:
        response = client.generate_icons(state["revised_sentences"])
        state["request_id"] = response.request_id
        state["icons"] = response.icons
        state["image_data"] = {}

        # Build the results display with images and sentences
        results_html = f'<div class="results-container"><h2>{state["title"]}</h2>'

        for icon in response.icons:
            image_id = icon.image_path.split("/")[-1] if icon.image_path else ""

            # Fetch the actual image
            try:
                img_bytes = client.fetch_icon(response.request_id, image_id)
                state["image_data"][image_id] = img_bytes

                import base64
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                img_src = f"data:image/png;base64,{img_b64}"
            except Exception:
                img_src = ""

            highlight_class = "highlighted" if icon.highlighted else ""
            highlight_badge = (
                '<span class="badge">Key Point</span>' if icon.highlighted else ""
            )

            results_html += f"""
            <div class="sentence-row {highlight_class}">
                <div class="icon-col">
                    {'<img src="' + img_src + '" alt="icon" />' if img_src else '<div class="placeholder">Loading...</div>'}
                </div>
                <div class="text-col">
                    {highlight_badge}
                    <p>{icon.sentence}</p>
                </div>
            </div>
            """

        results_html += "</div>"

        return (
            gr.update(visible=False),  # error
            gr.update(visible=True),   # results section
            gr.update(visible=True, value=results_html),
            gr.update(visible=True),   # export row
            None,  # docx download
            None,  # md download
        )

    except Exception:
        return (
            gr.update(visible=True, value="Failed to generate icons. Please try again."),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
        )


def export_docx():
    """Export current results to DOCX format."""
    if not state["icons"]:
        return None

    sentences = [
        {
            "sentence": icon.sentence,
            "image_prompt": icon.image_prompt,
            "highlighted": icon.highlighted,
            "image_path": icon.image_path,
        }
        for icon in state["icons"]
    ]

    docx_bytes = export_to_docx_bytes(
        title=state["title"],
        sentences=sentences,
        image_data=state["image_data"],
    )

    with tempfile.NamedTemporaryFile(
        suffix=".docx", prefix="easyread_", delete=False
    ) as f:
        f.write(docx_bytes)
        return f.name


def export_markdown():
    """Export current results to Markdown format."""
    if not state["icons"]:
        return None

    sentences = [
        {
            "sentence": icon.sentence,
            "image_prompt": icon.image_prompt,
            "highlighted": icon.highlighted,
            "image_path": icon.image_path,
        }
        for icon in state["icons"]
    ]

    md_bytes = export_to_markdown_bytes(
        title=state["title"],
        sentences=sentences,
        include_images=True,
        image_base_url=client.base_url,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".md", prefix="easyread_", delete=False
    ) as f:
        f.write(md_bytes)
        return f.name


def reset_app():
    """Reset the application state."""
    state.update({
        "title": "",
        "revised_sentences": [],
        "request_id": "",
        "icons": [],
        "image_data": {},
    })
    return (
        "",   # text input
        "",   # context
        "",   # terms
        gr.update(visible=False),  # error
        gr.update(visible=False),  # review section
        gr.update(visible=False),  # results section
    )


# Custom CSS
custom_css = """
.container { max-width: 900px; margin: auto; }
.section-box {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    background: #fafafa;
}
.section-title {
    font-size: 1.2em;
    font-weight: 600;
    color: #333;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid #0066cc;
}
.results-container {
    max-width: 100%;
    font-family: Arial, sans-serif;
}
.results-container h2 {
    text-align: center;
    color: #333;
    margin-bottom: 25px;
    padding-bottom: 12px;
    border-bottom: 2px solid #0066cc;
}
.sentence-row {
    display: flex;
    align-items: center;
    padding: 15px;
    margin-bottom: 12px;
    background: white;
    border-radius: 10px;
    border-left: 4px solid #ddd;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.sentence-row.highlighted {
    background: #e6f3ff;
    border-left: 4px solid #0066cc;
}
.icon-col {
    flex: 0 0 100px;
    text-align: center;
    margin-right: 20px;
}
.icon-col img {
    max-width: 80px;
    max-height: 80px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.icon-col .placeholder {
    width: 80px;
    height: 80px;
    background: #eee;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #999;
    font-size: 12px;
}
.text-col {
    flex: 1;
}
.text-col p {
    margin: 0;
    font-size: 16px;
    line-height: 1.5;
    color: #333;
}
.badge {
    display: inline-block;
    background: #0066cc;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    margin-bottom: 6px;
}
.error-box {
    background: #fee;
    border: 1px solid #fcc;
    color: #c00;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 15px;
}
"""

# Build the Gradio interface
with gr.Blocks(
    title="EasyRead - Document Simplification",
    theme=gr.themes.Soft(),
    css=custom_css,
) as app:

    gr.Markdown(
        """
        # EasyRead Document Simplification

        Convert complex documents into Easy Read format with pictogram icons.
        """
    )

    # Step 1: Input Section (always visible)
    with gr.Group():
        gr.Markdown("### Step 1: Enter Your Text")

        input_text = gr.Textbox(
            label="Text to Simplify",
            placeholder="Paste or type the text you want to convert to Easy Read format...",
            lines=8,
        )

        with gr.Row():
            context_input = gr.Textbox(
                label="Custom Context (Optional)",
                placeholder="Add context to help with simplification...",
                lines=2,
                scale=1,
            )
            terms_input = gr.Textbox(
                label="Terms to Preserve (Optional)",
                placeholder="Comma-separated terms...",
                lines=2,
                scale=1,
            )

        with gr.Row():
            simplify_btn = gr.Button("Simplify Text", variant="primary", size="lg")
            reset_btn = gr.Button("Reset", variant="secondary")

    # Error display
    error_box = gr.Markdown(visible=False, elem_classes=["error-box"])

    # Step 2: Review Section (appears after simplification)
    with gr.Group(visible=False, elem_classes=["section-box"]) as review_section:
        gr.Markdown("### Step 2: Review & Approve Sentences")

        title_display = gr.Markdown()

        with gr.Accordion("Validation Feedback", open=False):
            validation_display = gr.Markdown()

        gr.Markdown(
            "Edit the sentences below if needed, then click **Generate Icons**.",
            elem_classes=["hint"],
        )

        sentences_table = gr.Dataframe(
            headers=["#", "Sentence", "Image Prompt", "Highlighted"],
            datatype=["number", "str", "str", "bool"],
            col_count=(4, "fixed"),
            interactive=True,
            wrap=True,
        )

        generate_btn = gr.Button(
            "Generate Icons",
            variant="primary",
            size="lg",
        )

    # Step 3: Results Section (appears after icon generation)
    with gr.Group(visible=False, elem_classes=["section-box"]) as results_section:
        gr.Markdown("### Step 3: Your Easy Read Document")

        results_display = gr.HTML()

        with gr.Row(visible=False) as export_row:
            docx_btn = gr.Button("Export to Word (.docx)", variant="secondary")
            md_btn = gr.Button("Export to Markdown (.md)", variant="secondary")

        with gr.Row():
            docx_download = gr.File(label="Word Document", visible=False)
            md_download = gr.File(label="Markdown File", visible=False)

    # Event handlers
    simplify_btn.click(
        fn=simplify_text,
        inputs=[input_text, context_input, terms_input],
        outputs=[
            error_box,
            review_section,
            results_section,
            title_display,
            validation_display,
            sentences_table,
        ],
    )

    generate_btn.click(
        fn=generate_icons,
        inputs=[sentences_table],
        outputs=[
            error_box,
            results_section,
            results_display,
            export_row,
            docx_download,
            md_download,
        ],
    )

    docx_btn.click(
        fn=export_docx,
        inputs=[],
        outputs=[docx_download],
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[docx_download],
    )

    md_btn.click(
        fn=export_markdown,
        inputs=[],
        outputs=[md_download],
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[md_download],
    )

    reset_btn.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            input_text,
            context_input,
            terms_input,
            error_box,
            review_section,
            results_section,
        ],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False)
