import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

import gradio as gr
import torch

from NatureLM.config import Config
from NatureLM.models.NatureLM import NatureLM
from NatureLM.utils import generate_sample_batches, prepare_sample_waveforms

CONFIG: Config = None
MODEL: NatureLM = None


def prompt_lm(audios: list[str], messages: list[dict[str, str]]):
    cuda_enabled = torch.cuda.is_available()
    samples = prepare_sample_waveforms(audios, cuda_enabled)
    prompt_text = MODEL.llama_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).removeprefix(MODEL.llama_tokenizer.bos_token)

    prompt_text = re.sub(
        r"<\|start_header_id\|>system<\|end_header_id\|>\n\nCutting Knowledge Date: [^\n]+\nToday Date: [^\n]+\n\n<\|eot_id\|>",
        "",
        prompt_text,
    )  # exclude the system header from the prompt
    prompt_text = re.sub("\\n", r"\\n", prompt_text)  # FIXME this is a hack to fix the issue #34

    print(f"{prompt_text=}")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_answer = MODEL.generate(samples, CONFIG.generate, prompts=[prompt_text])
    return llm_answer[0]


def _multimodal_textbox_factory():
    return gr.MultimodalTextbox(
        value=None,
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        submit_btn="Add input",
        file_types=["audio"],
    )


def user_message(content):
    return {"role": "user", "content": content}


def add_message(history, message):
    for x in message["files"]:
        history.append(user_message({"path": x}))
    if message["text"]:
        history.append(user_message(message["text"]))
    return history, _multimodal_textbox_factory()


def combine_model_inputs(msgs: list[dict[str, str]]) -> dict[str, list[str]]:
    messages = []
    files = []
    for msg in msgs:
        print(msg, messages, files)
        match msg:
            case {"content": (path,)}:
                messages.append({"role": msg["role"], "content": "<Audio><AudioHere></Audio> "})
                files.append(path)
            case _:
                messages.append(msg)
    joined_messages = []
    # join consecutive messages from the same role
    for msg in messages:
        if joined_messages and joined_messages[-1]["role"] == msg["role"]:
            joined_messages[-1]["content"] += msg["content"]
        else:
            joined_messages.append(msg)

    return {"messages": joined_messages, "files": files}


def bot_response(history: list):
    print(type(history))
    combined_inputs = combine_model_inputs(history)
    response = prompt_lm(combined_inputs["files"], combined_inputs["messages"])
    history.append({"role": "assistant", "content": response})

    return history


def _chat_tab(examples):
    chatbot = gr.Chatbot(
        label="Model inputs",
        elem_id="chatbot",
        bubble_full_width=False,
        type="messages",
        render_markdown=False,
        # editable="user",  # disable because of https://github.com/gradio-app/gradio/issues/10320
        resizeable=True,
    )

    chat_input = _multimodal_textbox_factory()
    send_all = gr.Button("Send all", elem_id="send-all")
    clear_button = gr.ClearButton(components=[chatbot, chat_input], visible=False)

    chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = send_all.click(
        bot_response,
        [chatbot],
        [chatbot],
        api_name="bot_response",
    )

    bot_msg.then(lambda: gr.ClearButton(visible=True), None, [clear_button])
    clear_button.click(lambda: gr.ClearButton(visible=False), None, [clear_button])

    gr.Examples(
        list(examples.values()),
        chatbot,
        chatbot,
        example_labels=list(examples.keys()),
        examples_per_page=20,
    )


def summarize_batch_results(results):
    summary = Counter(results)
    summary_str = "\n".join(f"{k}: {v}" for k, v in summary.most_common())
    return summary_str


def run_batch_inference(files, task, progress=gr.Progress()) -> str:
    outputs = []
    prompt = "<Audio><AudioHere></Audio> " + task

    for file in progress.tqdm(files):
        outputs.append(prompt_lm([file], [{"role": "user", "content": prompt}]))

    batch_summary: str = summarize_batch_results(outputs)
    report = f"Batch summary:\n{batch_summary}\n\n"
    return report


def multi_extension_glob_mask(mask_base, *extensions):
    mask_ext = ["[{}]".format("".join(set(c))) for c in zip(*extensions)]
    if not mask_ext or len(set(len(e) for e in extensions)) > 1:
        mask_ext.append("*")
    return mask_base + "".join(mask_ext)


def _batch_tab(file_selection: Literal["upload", "explorer"] = "upload"):
    if file_selection == "explorer":
        files = gr.FileExplorer(
            glob=multi_extension_glob_mask("**.", "mp3", "flac", "wav"),
            label="Select audio files",
            file_count="multiple",
        )
    elif file_selection == "upload":
        files = gr.Files(label="Uploaded files", file_types=["audio"], height=300)
    task = gr.Textbox(label="Task", placeholder="Enter task...", show_label=True)

    process_btn = gr.Button("Process")
    output = gr.TextArea()

    process_btn.click(
        run_batch_inference,
        [files, task],
        [output],
    )


def to_raven_format(outputs: dict[int, str], chunk_len: int = 10) -> str:
    def get_line(row, start, end, annotation):
        return f"{row}\tSpectrogram 1\t1\t{start}\t{end}\t0\t8000\t{annotation}"

    raven_output = ["Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation"]
    current_offset = 0
    last_label = ""
    row = 1

    # The "Selection" column is just the row number.
    # The "view" column will always say "Spectrogram 1".
    # Channel can always be "1".
    # For the frequency bounds we can just use 0 and 1/2 the sample rate
    for offset, label in sorted(outputs.items()):
        if label != last_label and last_label:
            raven_output.append(get_line(row, current_offset, offset, last_label))
            current_offset = offset
            row += 1
        if not last_label:
            current_offset = offset
        if label != "None":
            last_label = label
        else:
            last_label = ""
    if last_label:
        raven_output.append(get_line(row, current_offset, current_offset + chunk_len, last_label))

    return "\n".join(raven_output)


def _run_long_recording_inference(file, task, chunk_len: int = 10, hop_len: int = 5, progress=gr.Progress()):
    cuda_enabled = torch.cuda.is_available()
    outputs = {}
    offset = 0

    prompt = f"<Audio><AudioHere></Audio> {task}"
    prompt = CONFIG.model.prompt_template.format(prompt)

    for batch in progress.tqdm(generate_sample_batches(file, cuda_enabled, chunk_len=chunk_len, hop_len=hop_len)):
        prompt_strs = [prompt] * len(batch["audio_chunk_sizes"])
        with torch.cuda.amp.autocast(dtype=torch.float16):
            llm_answers = MODEL.generate(batch, CONFIG.generate, prompts=prompt_strs)
        for answer in llm_answers:
            outputs[offset] = answer
            offset += hop_len

    report = f"Number of chunks: {len(outputs)}\n\n"
    for offset in sorted(outputs.keys()):
        report += f"{offset:02d}s:\t{outputs[offset]}\n"

    raven_output = to_raven_format(outputs, chunk_len=chunk_len)
    with tempfile.NamedTemporaryFile(mode="w", prefix="raven-", suffix=".txt", delete=False) as f:
        f.write(raven_output)
        raven_file = f.name

    return report, raven_file


def _long_recording_tab():
    audio_input = gr.Audio(label="Upload audio file", type="filepath")
    task = gr.Dropdown(
        [
            "What are the common names for the species in the audio, if any?",
            "Caption the audio.",
            "Caption the audio, using the scientific name for any animal species.",
            "Caption the audio, using the common name for any animal species.",
            "What is the scientific name for the focal species in the audio?",
            "What is the common name for the focal species in the audio?",
            "What is the family of the focal species in the audio?",
            "What is the genus of the focal species in the audio?",
            "What is the taxonomic name of the focal species in the audio?",
            "What call types are heard from the focal species in the audio?",
            "What is the life stage of the focal species in the audio?",
        ],
        label="Tasks",
        allow_custom_value=True,
    )
    with gr.Accordion("Advanced options", open=False):
        hop_len = gr.Slider(1, 10, 5, label="Hop length (seconds)", step=1)
        chunk_len = gr.Slider(1, 10, 10, label="Chunk length (seconds)", step=1)
    process_btn = gr.Button("Process")
    output = gr.TextArea()
    download_raven = gr.DownloadButton("Download Raven file")

    process_btn.click(
        _run_long_recording_inference,
        [audio_input, task, chunk_len, hop_len],
        [output, download_raven],
    )


def main(
    assets_dir: Path,
    port: int,
    show_errors: bool,
    cfg_path: str | Path,
    options: list[str] = [],
    device: str = "cuda:0",
):
    cfg = Config.from_sources(yaml_file=cfg_path, cli_args=options)
    model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio", force_download=True)
    model.to(device)
    model.eval()

    global MODEL, CONFIG
    MODEL = model
    CONFIG = cfg

    laz_audio = assets_dir / "Lazuli_Bunting_yell-YELLLAZB20160625SM303143.mp3"
    frog_audio = assets_dir / "nri-GreenTreeFrogEvergladesNP.mp3"
    robin_audio = assets_dir / "yell-YELLAMRO20160506SM3.mp3"
    vireo_audio = assets_dir / "yell-YELLWarblingVireoMammoth20150614T29ms.mp3"

    examples = {
        "Caption the audio (Lazuli Bunting)": [
            [
                user_message({"path": str(laz_audio)}),
                user_message("Caption the audio."),
            ]
        ],
        "Caption the audio (Green Tree Frog)": [
            [
                user_message({"path": str(frog_audio)}),
                user_message("Caption the audio, using the common name for any animal species."),
            ]
        ],
        "Caption the audio (American Robin)": [
            [
                user_message({"path": str(robin_audio)}),
                user_message("Caption the audio."),
            ]
        ],
        "Caption the audio (Warbling Vireo)": [
            [
                user_message({"path": str(vireo_audio)}),
                user_message("Caption the audio."),
            ]
        ],
    }

    with gr.Blocks(title="NatureLM-audio", theme=gr.themes.Default(primary_hue="slate")) as app:
        with gr.Tabs():
            with gr.Tab("Chat"):
                _chat_tab(examples)
            with gr.Tab("Batch"):
                _batch_tab()
            with gr.Tab("Long Recording"):
                _long_recording_tab()

    app.launch(
        server_port=port,
        server_name="0.0.0.0",
        auth=None,
        show_error=show_errors,
        favicon_path=str(assets_dir / "esp_favicon.png"),
    )
