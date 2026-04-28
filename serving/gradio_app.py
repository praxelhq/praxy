"""Praxy Voice — Gradio app deployed on Modal.

Live at:  https://<workspace>--praxy-voice-demo-voice.modal.run/
Custom:   voice.praxel.in (CNAME — see docs/deploy_voice_praxel.md)

Architecture: this app is a thin Gradio frontend that dispatches to the
already-deployed inference classes in `serving.modal_app`:
    PraxyChatterboxLoRA  (R6 LoRA, pure Te / Ta)
    IndicF5TTS           (Hi pure + all code-mix via translit)

Both classes are looked up cross-app via `modal.Cls.from_name`, so this
file does NOT re-build any TTS image. Image here is CPU-only and tiny,
keeping cold start < 5s.

Deploy:
    modal deploy serving/modal_app.py        # ensure backends are deployed
    modal deploy serving/gradio_app.py       # then ship the frontend

Local dev (hot-reload):
    modal serve serving/gradio_app.py
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "praxy-voice-demo"

app = modal.App(APP_NAME)

# Lightweight CPU image — just Gradio + the bits we need to run on the
# frontend (text preprocess, codemix detection, Anthropic call).
demo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio==4.44.0",  # 5.x has Modal/svelte URL-construction quirks
        "huggingface_hub<1.0",  # gradio 4.44 imports HfFolder, removed in HF 1.x
        "fastapi",
        "uvicorn",
        "requests",  # gradio.cli imports this at gradio package import time
        "soundfile==0.12.1",
        "numpy",
        "anthropic",
        "indic-transliteration",
    )
    .add_local_dir(str(REPO_ROOT / "serving"), "/repo/serving", copy=True)
    .add_local_dir(str(REPO_ROOT / "praxy"), "/repo/praxy", copy=True)
    .add_local_dir(str(REPO_ROOT / "evaluation"), "/repo/evaluation", copy=True)
    .add_local_dir(str(REPO_ROOT / "data" / "references"), "/repo/data/references", copy=True)
    .env({"PYTHONPATH": "/repo"})
)


# Voice library — pre-made reference voices bundled with the demo.
# Reference paths are resolved relative to /repo inside the container.
VOICE_LIBRARY: dict[str, dict] = {
    "Hindi · Female (default)": {
        "lang": "hi",
        "ref_audio": "/repo/data/references/sarvam_hi_female_10s.wav",
        "ref_text": "मेरे दादा जी हर शाम बरामदे में बैठकर अपने बचपन की कहानियां सुनाते हैं और हम सब मिलकर बडे चाव से उनकी बातें सुनते हैं",
    },
    "Hindi · Female (alt)": {
        "lang": "hi",
        "ref_audio": "/repo/data/references/cartesia_hi_female_6s.wav",
        "ref_text": "मेरे दादाजी हर शाम बरामदे में बैठकर अपने बचपन की कहानियां सुनाते हैं और हम सब मिलकर बड़े चाव से उनकी बातें सुनते हैं",
    },
    "Telugu · Female": {
        "lang": "te",
        "ref_audio": "/repo/data/references/sarvam_te_female_9s.wav",
        "ref_text": "మా తాతయ్య ప్రతి సాయంత్రం వరండాలో కూర్చుని తన చిన్నతనంలో జరిగిన కథలు చెబుతూ ఉంటారు మరియు మేము అందరూ కలిసి ఆసక్తిగా వినేవాళ్లం",
    },
    "Tamil · Male": {
        "lang": "ta",
        "ref_audio": "/repo/data/references/sarvam_ta_male_11s.wav",
        "ref_text": "எங்கள் தாத்தா தினமும் மாலையில் திண்ணையில் அமர்ந்து கொண்டு தன் சிறுவயதில் நடந்த கதைகளைச் சொல்லிக் கொண்டிருப்பார் நாங்கள் அனைவரும் சேர்ந்து ஆர்வமாகக் கேட்போம்",
    },
}

LANG_CODE_TO_NAME = {"hi": "Hindi", "te": "Telugu", "ta": "Tamil", "en": "English"}

EXAMPLES = [
    ["Hindi · Female (default)", "नमस्ते! मेरा नाम प्राक्सी है, और मैं भारत के लिए बनाई गई एक ओपन-सोर्स आवाज़ हूँ।"],
    ["Hindi · Female (default)", "मैंने WhatsApp पे message किया but notification नहीं आया।"],
    ["Telugu · Female", "నమస్తే! నేను ప్రాక్సీ, ఇండియా కోసం రూపొందించిన ఓపెన్-సోర్స్ వాయిస్."],
    ["Telugu · Female", "మా CEO ఇవాళ all-hands meeting లో కొత్త quarterly targets announce చేశారు."],
    ["Tamil · Male", "வணக்கம்! நான் ப்ராக்ஸி, இந்தியாவுக்காக உருவாக்கப்பட்ட ஓபன்-சோர்ஸ் குரல்."],
]


def _detect_lang(text: str) -> str:
    """Auto-detect target language from script. Used when user uploads a
    custom voice (we don't ask the lang explicitly to keep UI simple)."""
    if re.search(r"[ఀ-౿]", text):
        return "te"
    if re.search(r"[஀-௿]", text):
        return "ta"
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    return "en"


def _is_codemix(text: str, lang: str) -> bool:
    """Same rule as serving.praxy_router.is_codemix — Latin word ≥2 chars
    inside a non-English target."""
    return bool(re.search(r"[A-Za-z]{2,}", text)) and lang != "en"


@app.function(
    image=demo_image,
    secrets=[
        modal.Secret.from_name("praxy-hf"),
        modal.Secret.from_name("praxy-anthropic"),
    ],
    timeout=900,
    min_containers=1,
    max_containers=4,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="voice")
def gradio_asgi():
    """ASGI app exposing a Gradio UI. CPU-only; the synth itself is
    offloaded cross-app to the deployed praxy-voice classes."""
    import gradio as gr
    import soundfile as sf
    import numpy as np
    from fastapi import FastAPI

    # Cross-app class lookups. praxy-voice must be deployed first.
    PraxyChatterboxLoRA = modal.Cls.from_name("praxy-voice", "PraxyChatterboxLoRA")
    IndicF5TTS = modal.Cls.from_name("praxy-voice", "IndicF5TTS")

    # Pre-instantiate. Each class call routes to its already-deployed
    # container; .with_options() lets us inject env per-call.
    indicf5 = IndicF5TTS()

    def synth(text: str, voice_choice: str, custom_audio_path, custom_text: str):
        if not text or not text.strip():
            return None, "❌ Please type some text to synthesise."

        custom_mode = voice_choice.startswith("📤")
        if custom_mode and custom_audio_path is None:
            return None, "❌ Pick a pre-made voice or upload a reference clip."

        if custom_mode:
            ref_audio_path = custom_audio_path
            ref_text = (custom_text or "").strip()
            if not ref_text:
                return None, "❌ Custom voice needs a reference transcript matching the audio."
            lang = _detect_lang(text)
            with open(ref_audio_path, "rb") as f:
                ref_bytes = f.read()
        else:
            voice = VOICE_LIBRARY[voice_choice]
            try:
                with open(voice["ref_audio"], "rb") as f:
                    ref_bytes = f.read()
            except FileNotFoundError:
                return None, f"❌ Bundled ref not found at {voice['ref_audio']!r} (deploy bug)."
            ref_text = voice["ref_text"]
            lang = voice["lang"]

        codemix = _is_codemix(text, lang)

        # 1. Codemix path: transliterate + IndicF5
        if codemix:
            try:
                from serving.codemix_to_native_script import transliterate_codemix
                text_for_synth = transliterate_codemix(text, lang)
            except Exception as e:
                return None, f"❌ Transliteration failed: {type(e).__name__}: {e}"
            t0 = time.time()
            try:
                wav_bytes, sr = indicf5.synthesize.remote(
                    text=text_for_synth,
                    ref_audio_bytes=ref_bytes,
                    ref_text=ref_text,
                )
            except Exception as e:
                return None, f"❌ IndicF5 synth failed: {type(e).__name__}: {e}"
            out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out.write(wav_bytes); out.close()
            return out.name, (
                f"✓ {LANG_CODE_TO_NAME.get(lang, lang)} code-mix · "
                f"transliterate→IndicF5 · {time.time()-t0:.1f}s"
            )

        # 2. Pure-Hi path: IndicF5 zero-shot also handles Hi pure decently
        # (WER 0.13 vs vanilla Chatterbox 0.025; vanilla wins but IndicF5
        # ships with the same call shape, simpler routing here)
        if lang in ("hi", "ta"):
            t0 = time.time()
            try:
                wav_bytes, sr = indicf5.synthesize.remote(
                    text=text,
                    ref_audio_bytes=ref_bytes,
                    ref_text=ref_text,
                )
            except Exception as e:
                return None, f"❌ IndicF5 synth failed: {type(e).__name__}: {e}"
            out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out.write(wav_bytes); out.close()
            return out.name, (
                f"✓ {LANG_CODE_TO_NAME.get(lang, lang)} pure · "
                f"IndicF5 zero-shot · {time.time()-t0:.1f}s"
            )

        # 3. Te pure & En pure → Chatterbox + R6 LoRA branch.
        env = {
            "PRAXY_CKPT_PATH": "/cache/chatterbox_indic/round_6/step_8000.ckpt",
            "PRAXY_USE_BUPS": "1" if lang == "te" else "0",
            "PRAXY_NO_LORA": "1" if lang == "en" else "0",
        }
        synth_cls = PraxyChatterboxLoRA.with_options(
            secrets=[modal.Secret.from_dict(env), modal.Secret.from_name("praxy-hf")]
        )()
        t0 = time.time()
        try:
            wav_bytes, sr = synth_cls.synthesize.remote(
                text=text,
                language_code=lang,
                ref_audio_bytes=ref_bytes,
                exaggeration=0.7,
                cfg_weight=0.5,
                temperature=0.6,
                repetition_penalty=2.0,
                min_p=0.1,
                top_p=1.0,
            )
        except Exception as e:
            return None, f"❌ Chatterbox synth failed: {type(e).__name__}: {e}"
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.write(wav_bytes); out.close()
        branch = "vanilla Chatterbox" if lang == "en" else "Chatterbox + R6 LoRA"
        return out.name, (
            f"✓ {LANG_CODE_TO_NAME.get(lang, lang)} · {branch} · "
            f"{time.time()-t0:.1f}s"
        )

    css = """
    .gradio-container { max-width: 980px !important; }
    h1 { font-weight: 700; margin-bottom: 0.4em; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Praxy Voice", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            "# 🎤 Praxy Voice\n"
            "Open-source Hindi · Telugu · Tamil · English TTS, including code-mix. "
            "Voice cloning from any 8–15 s reference clip. "
            "Built on [Chatterbox](https://github.com/resemble-ai/chatterbox) + "
            "[IndicF5](https://huggingface.co/ai4bharat/IndicF5) + a Haiku-driven "
            "native-script transliteration preprocessor.\n\n"
            "*v1 · Code: [github.com/praxelhq/praxy](https://github.com/praxelhq/praxy) · "
            "Paper: arXiv (link soon).*"
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    label="Text to synthesise",
                    placeholder="मैंने WhatsApp पे message किया but notification नहीं आया।",
                    lines=4,
                )
                voice_in = gr.Dropdown(
                    list(VOICE_LIBRARY.keys()) + ["📤 Use custom upload below"],
                    value=list(VOICE_LIBRARY.keys())[0],
                    label="Voice",
                )
                with gr.Accordion("📤 Or clone your own voice (8–15 s clip)", open=False):
                    custom_audio = gr.Audio(label="Reference audio", type="filepath")
                    custom_text = gr.Textbox(
                        label="Reference transcript (must match the audio exactly, in the script you want to synthesise)",
                        lines=2,
                    )
                btn = gr.Button("Generate", variant="primary", size="lg")
            with gr.Column(scale=1):
                audio_out = gr.Audio(label="Output", autoplay=True)
                status = gr.Markdown("")

        gr.Examples(EXAMPLES, inputs=[voice_in, text_in], label="Try one of these")

        btn.click(
            synth,
            inputs=[text_in, voice_in, custom_audio, custom_text],
            outputs=[audio_out, status],
        )

        gr.Markdown(
            "---\n"
            "**Code-mix** (Latin English words inside Indic text) is auto-detected and "
            "transliterated to native script before synth — that's why *WhatsApp* gets "
            "pronounced *vaa-ts-ay-p* rather than American *whats-app*. It matches how "
            "Indians actually code-switch.\n\n"
            "**Privacy**: uploaded reference clips are not logged or stored."
        )

    demo.queue(max_size=10)
    fastapi_app = FastAPI(title="Praxy Voice")
    return gr.mount_gradio_app(fastapi_app, demo, path="/")
