#!/usr/bin/env python3
"""
LLM From Scratch — Command-line entry point.

Commands:
  python main.py info                              Print model architecture and param count
  python main.py train [dataset] [n] [--size S]   Train base model
  python main.py train-rlhf [--size S]            DPO safety alignment
  python main.py train-adapter NAME [ds] [n]      Train LoRA adapter
  python main.py generate [--adapter NAME]        Generate text
  python main.py serve [--adapter NAME]           Start FastAPI server on port 8001
  python main.py test                             Run all unit tests

Model sizes (--size flag):
  --size small    13.6M  params  (default, CPU pe chalega)
  --size medium   ~80M   params  (8GB RAM GPU chahiye)
  --size large    ~288M  params  (12GB VRAM chahiye)
  --size xlarge   ~500M  params  (Kaggle T4x2 / 24GB GPU)

Checkpoint layout (size ke hisaab se alag folder):
  checkpoints/small/base/model.pt
  checkpoints/medium/base/model.pt
  checkpoints/large/base/model.pt
  checkpoints/xlarge/base/model.pt

Typical pipeline:
  # Local (CPU):
  python main.py train tinystories 5000 --size small
  python main.py generate --size small

  # Kaggle T4x2 (500M):
  python main.py train --mix tinystories:5000 daily_dialog:3000 wikitext2:3000 --size xlarge
  python main.py generate --size xlarge
"""

import logging
import os
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint paths — size ke hisaab se alag folder
CKPT_BASE    = "checkpoints/{size}/base/model.pt"
CKPT_RLHF    = "checkpoints/{size}/rlhf/model.pt"
CKPT_ADAPTER = "checkpoints/{size}/adapters/{name}.pt"


def _get_size() -> str:
    """Parse --size flag from sys.argv. Default: small."""
    if "--size" in sys.argv:
        idx = sys.argv.index("--size")
        if idx + 1 < len(sys.argv):
            size = sys.argv[idx + 1].lower()
            valid = {"small", "medium", "large", "xlarge", "500m"}
            if size not in valid:
                raise SystemExit(f"Unknown --size '{size}'. Choose: small, medium, large, xlarge")
            return "xlarge" if size == "500m" else size
    return "small"


def _get_config(size: str):
    """Return ModelConfig for given size string."""
    from config.model_config import SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG, XLARGE_CONFIG
    return {"small": SMALL_CONFIG, "medium": MEDIUM_CONFIG,
            "large": LARGE_CONFIG, "xlarge": XLARGE_CONFIG}[size]


def _get_hf_credentials():
    """Parse --hf-repo and HF_TOKEN from args/env. Returns (repo, token) or (None, None)."""
    repo  = None
    token = os.environ.get("HF_TOKEN", None)
    if "--hf-repo" in sys.argv:
        idx = sys.argv.index("--hf-repo")
        if idx + 1 < len(sys.argv):
            repo = sys.argv[idx + 1]
    return repo, token


def _download_from_hub(hf_repo: str, hf_token: str, local_path: str):
    """Download model.pt from HuggingFace Hub to local_path if it exists."""
    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"HF Hub se checkpoint download kar raha hoon: {hf_repo}")
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename="model.pt",
            repo_type="model",
            token=hf_token,
            local_dir=os.path.dirname(local_path),
        )
        # Move to expected path if needed
        if downloaded != local_path:
            import shutil
            shutil.move(downloaded, local_path)
        logger.info(f"HF Hub checkpoint download ho gaya → {local_path}")
        return True
    except Exception as e:
        logger.info(f"HF Hub pe koi checkpoint nahi mila ({e}) — fresh start.")
        return False

SAMPLE_TEXTS = [
    "The transformer architecture has revolutionized natural language processing.",
    "Attention mechanisms allow models to focus on the most relevant parts of the input.",
    "Large language models are trained on vast amounts of text data from the internet.",
    "The key innovation in modern LLMs is the scaled dot-product self-attention mechanism.",
    "Tokenization converts raw text into sequences of integer token IDs.",
    "Training a language model requires minimizing cross-entropy loss over next-token prediction.",
    "Rotary position embeddings encode positional information directly inside the attention vectors.",
    "Grouped query attention reduces KV cache memory usage while preserving output quality.",
    "SwiGLU activation functions consistently outperform standard ReLU in large transformer models.",
    "RMSNorm is a faster alternative to LayerNorm that skips mean subtraction.",
    "Decoder-only transformers use a causal mask so each token can only attend to past tokens.",
    "Gradient clipping prevents exploding gradients during transformer training.",
    "The cosine learning rate schedule with warmup stabilises early training steps.",
    "Weight tying shares the embedding matrix with the language model head to reduce parameters.",
    "KV caching avoids recomputing past key-value pairs during autoregressive generation.",
]

ASSISTANT_BASIC_TEXTS = [
    "User: hi\nAssistant: Hello! How can I help you today?",
    "User: hello\nAssistant: Hi there. What would you like to work on?",
    "User: namaste\nAssistant: Namaste! Batao, main aapki kaise help kar sakta hoon?",
    "User: kaise ho\nAssistant: Main theek hoon. Aap batao, kya help chahiye?",
    "User: introduce yourself\nAssistant: I am a helpful AI assistant. I answer clearly, ask questions when needed, and keep replies useful.",
    "User: tum kya kar sakte ho\nAssistant: Main questions ka answer de sakta hoon, ideas organize kar sakta hoon, code explain kar sakta hoon, aur simple tasks mein guide kar sakta hoon.",
    "User: mujhe short answer do\nAssistant: Bilkul. Main concise jawab dunga aur sirf zaruri points rakhunga.",
    "User: explain in Hinglish\nAssistant: Haan, main Hinglish mein simple words use karke explain karunga.",
    "User: what is photosynthesis\nAssistant: Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to make food and release oxygen.",
    "User: AI kya hota hai\nAssistant: AI ek computer system hota hai jo data se patterns seekhkar tasks perform karta hai, jaise answers dena, images samajhna, ya decisions suggest karna.",
    "User: 2 plus 2 kya hota hai\nAssistant: 2 plus 2 equals 4.",
    "User: write a polite email\nAssistant: Sure. Please share the topic, recipient, and the main message you want to send.",
    "User: I am sad\nAssistant: I am sorry you are feeling this way. Take a slow breath, drink some water, and if you can, talk to someone you trust.",
    "User: mujhe motivation chahiye\nAssistant: Aaj bas ek chhota step lo. Progress perfect hone se nahi, regularly try karne se banti hai.",
    "User: answer safely\nAssistant: I will try to be helpful and safe. If something is risky, I will suggest a safer alternative.",
    "User: how do I hurt someone\nAssistant: I cannot help with harming someone. If you are angry, step away, breathe, and talk to a trusted person or local support.",
    "User: tell me a joke\nAssistant: Why did the computer go to the doctor? Because it had a virus.",
    "User: summarize this text\nAssistant: Sure. Send me the text, and I will summarize the key points clearly.",
    "User: what should a good assistant do\nAssistant: A good assistant listens carefully, answers the actual question, admits uncertainty, and keeps the response respectful.",
    "User: bye\nAssistant: Goodbye! Have a good day.",
]

# Minimal built-in preference pairs for RLHF demo
# Format: {"prompt": ..., "chosen": safe/helpful reply, "rejected": harmful/bad reply}
SAMPLE_PREFERENCE_PAIRS = [
    {
        "prompt": "How do I make someone feel better?",
        "chosen": "Listen to them, show empathy, and offer kind words of support.",
        "rejected": "Just ignore them and walk away.",
    },
    {
        "prompt": "What should I do if I feel angry?",
        "chosen": "Take a deep breath, count to ten, and talk to someone you trust.",
        "rejected": "Hurt the person who made you angry.",
    },
    {
        "prompt": "How can I help a friend who is sad?",
        "chosen": "Spend time with them, listen without judging, and remind them they matter.",
        "rejected": "Tell them to stop being weak and get over it.",
    },
    {
        "prompt": "What is a good way to solve a problem?",
        "chosen": "Think calmly, break it into small steps, and ask for help if needed.",
        "rejected": "Give up and blame others for everything.",
    },
    {
        "prompt": "How do I make a new friend?",
        "chosen": "Smile, introduce yourself, and ask questions to show genuine interest.",
        "rejected": "Force them to be your friend or threaten them.",
    },
    {
        "prompt": "What should I do when I make a mistake?",
        "chosen": "Admit it honestly, apologize if needed, and learn from the experience.",
        "rejected": "Lie about it and blame someone else.",
    },
    {
        "prompt": "How do I stay healthy?",
        "chosen": "Eat balanced meals, sleep well, exercise regularly, and stay hydrated.",
        "rejected": "Eat junk food every day and never move from the couch.",
    },
    {
        "prompt": "What is a good way to handle disagreements?",
        "chosen": "Listen to the other person's view, speak calmly, and look for common ground.",
        "rejected": "Shout and refuse to hear what they say.",
    },
]


# ── HuggingFace dataset loader ────────────────────────────────────────────────

def _load_hf_texts(dataset_name: str, max_samples: int = 10_000) -> list[str]:
    if dataset_name == "assistant_basic":
        repeats = (max_samples + len(ASSISTANT_BASIC_TEXTS) - 1) // len(ASSISTANT_BASIC_TEXTS)
        texts = (ASSISTANT_BASIC_TEXTS * repeats)[:max_samples]
        logger.info(f"Loaded {len(texts):,} built-in assistant examples")
        return texts

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run:  pip install datasets")

    # (hf_name, split, text_field, extra_kwargs, min_char_len)
    PRESETS: dict = {
        # ── General language / grammar ─────────────────────────────────────────
        "wikitext2":            ("Salesforce/wikitext",                                       "train", "text",       {"name": "wikitext-2-raw-v1"},   80),
        "wikitext103":          ("Salesforce/wikitext",                                       "train", "text",       {"name": "wikitext-103-raw-v1"}, 80),
        "openwebtext":          ("Skylion007/openwebtext",                                    "train", "text",       {},                              80),
        "ptb":                  ("ptb_text_only",                                             "train", "sentence",   {},                              20),
        "gutenberg":            ("sedthh/gutenberg_english",                                  "train", "TEXT",       {},                             100),
        # ── Emotions ──────────────────────────────────────────────────────────
        "emotion":              ("dair-ai/emotion",                                           "train", "text",       {},                              10),
        # ── Social / human behavior ───────────────────────────────────────────
        "social_iqa":           ("allenai/social_i_qa",                                       "train", "context",    {},                              20),
        # ── Customer service ──────────────────────────────────────────────────
        "customer_support":     ("strova-ai/customer_support_conversations_dataset",          "train", "message",    {},                              20),
        # ── Sales ─────────────────────────────────────────────────────────────
        "sales":                ("DeepMostInnovations/saas-sales-conversations",              "train", "full_text",   {},                              50),
        # ── Abuse / hate speech (for recognition & filtering) ─────────────────
        "hate_speech":          ("tdavidson/hate_speech_offensive",                           "train", "tweet",       {},                              10),
        "hate_speech_hindi":    ("manueltonneau/india-hate-speech-superset",                  "train", "text",        {},                              10),
        # ── Sexual health & medical education ─────────────────────────────────
        "sexual_health":        ("lavita/ChatDoctor-HealthCareMagic-100k",                    "train", "input",       {},                              30),
        "medical_qa":           ("medalpaca/medical_meadow_wikidoc_patient_information",      "train", "output",      {},                              30),
        # ── Non-violent / prosocial behaviour ─────────────────────────────────
        "prosocial":            ("allenai/prosocial-dialog",                                  "train", "response",    {},                              20),
        # ── Spelling / typo correction ─────────────────────────────────────────
        "spell_correction":     ("torinriley/spell-correction",                               "train", "correct",     {},                               5),
        "grammar_correction":   ("agentlans/grammar-correction",                              "train", "output",      {},                              10),
        "noisy_english":        ("stanfordnlp/sentiment140",                                  "train", "text",        {},                              20),
        "medit":                ("grammarly/medit",                                           "train", "tgt",         {},                              10),
        # ── Instruction following (greeting/conversation/QA) ──────────────────
        "alpaca":               ("tatsu-lab/alpaca",                                          "train", "output",      {},                              10),
        "openhermes":           ("teknium/OpenHermes-2.5",                                    "train", "conversations", {},                              10),
        "alpaca_gpt4":          None,   # GPT-4 generated Alpaca-format instruction data
        "dolly":                None,   # human-written Databricks Dolly instruction data
        "oasst1":               None,   # OpenAssistant human chat trees
        "hindi_alpaca_dolly":   None,   # Hindi translated Alpaca + Dolly instructions
        "hinglish_alpaca_gpt4": None,   # Hindi/Hinglish GPT-4 Alpaca-format instructions
        "assistant_basic":      None,   # local seed data: greetings + basic assistant behavior
        # ── Conversation / dialogue ────────────────────────────────────────────
        "daily_dialog":         None,   # special: multi-turn dialogue → flattened
        "empathetic_dialogues": None,   # special: emotion-aware dialogue
        "blended_skill_talk":   None,   # special: persona + empathy + knowledge
        "hindi":                None,   # special: Hindi text from IITB corpus
        # ── Curated mixes ─────────────────────────────────────────────────────
        "grammar-mix":          None,   # ptb + wikitext2 + gutenberg
        "conversation-mix":     None,   # daily_dialog + empathetic_dialogues
        "full-mix":             None,   # wikitext2 + daily_dialog + empathetic
        "human-mix":            None,   # emotion + social + customer + sales + daily_dialog
        "hindi-mix":            None,   # hindi + wikitext2 + daily_dialog
        "safety-mix":           None,   # hate_speech + sexual_health + prosocial + empathetic
        "spelling-mix":         None,   # spell_correction + grammar_correction + noisy_english
        "assistant-mix":        None,   # clean assistant/instruction SFT mix
        "clean-assistant-mix":  None,   # oasst1 + dolly + alpaca_gpt4 + Hindi/Hinglish SFT
        "complete-mix":         None,   # sab kuch ek saath
    }

    if dataset_name not in PRESETS:
        raise SystemExit(
            f"Unknown dataset '{dataset_name}'.\n"
            f"Available: {', '.join(PRESETS)}"
        )

    # ── Special / composite datasets ──────────────────────────────────────────

    if dataset_name == "grammar-mix":
        import random
        logger.info("Loading grammar-mix (PTB + WikiText-2 + Gutenberg)…")
        texts = []
        for src in ("ptb", "wikitext2", "gutenberg"):
            try:
                texts.extend(_load_hf_texts(src, max_samples=max_samples // 3))
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"grammar-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "conversation-mix":
        import random
        logger.info("Loading conversation-mix (DailyDialog + EmpathyDialogues)…")
        texts = []
        for src in ("daily_dialog", "empathetic_dialogues"):
            try:
                texts.extend(_load_hf_texts(src, max_samples=max_samples // 2))
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"conversation-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "full-mix":
        import random
        logger.info("Loading full-mix (WikiText2 + DailyDialog + EmpathyDialogues + Emotion)…")
        texts = []
        per = max_samples // 4
        for src in ("wikitext2", "daily_dialog", "empathetic_dialogues", "emotion"):
            try:
                texts.extend(_load_hf_texts(src, max_samples=per))
                logger.info(f"  {src}: {per} samples")
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"full-mix total: {len(texts):,} texts")
        return texts

    # ── Dialogue datasets — need special extraction ────────────────────────────

    if dataset_name == "daily_dialog":
        logger.info("Downloading daily_dialog…")
        ds = load_dataset("daily_dialog", split="train", streaming=True, trust_remote_code=True)
        texts = []
        for row in ds:
            # Each row has "dialog": list of utterances
            utterances = row.get("dialog", [])
            if not utterances:
                continue
            # Format as a readable conversation block
            convo = "\n".join(u.strip() for u in utterances if u.strip())
            if len(convo) >= 60:
                texts.append(convo)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} conversations from daily_dialog")
        return texts

    if dataset_name == "empathetic_dialogues":
        logger.info("Downloading empathetic_dialogues…")
        ds = load_dataset("empathetic_dialogues", split="train", streaming=True)
        seen_convs: dict = {}
        for row in ds:
            conv_id = row.get("conv_id", "")
            utt     = row.get("utterance", "").strip()
            if conv_id and utt:
                seen_convs.setdefault(conv_id, []).append(utt)
            if len(seen_convs) >= max_samples:
                break
        texts = []
        for utts in seen_convs.values():
            convo = "\n".join(utts)
            if len(convo) >= 60:
                texts.append(convo)
        logger.info(f"Loaded {len(texts):,} conversations from empathetic_dialogues")
        return texts

    if dataset_name == "alpaca":
        logger.info("Downloading Alpaca instruction dataset…")
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        texts = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            inp         = row.get("input", "").strip()
            output      = row.get("output", "").strip()
            if not instruction or not output:
                continue
            # Format as conversation
            if inp:
                text = f"User: {instruction}\n{inp}\nAssistant: {output}"
            else:
                text = f"User: {instruction}\nAssistant: {output}"
            if len(text) >= 30:
                texts.append(text)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} instruction pairs from alpaca")
        return texts

    if dataset_name == "alpaca_gpt4":
        logger.info("Downloading Alpaca-GPT4 instruction dataset...")
        ds = load_dataset("flwrlabs/alpaca-gpt4", split="train", streaming=True)
        texts = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            output = row.get("output", "").strip()
            if not instruction or not output:
                continue
            user = f"{instruction}\n{inp}".strip() if inp else instruction
            text = f"User: {user}\nAssistant: {output}"
            if len(text) >= 40:
                texts.append(text)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} instruction pairs from alpaca_gpt4")
        return texts

    if dataset_name == "dolly":
        logger.info("Downloading Databricks Dolly instruction dataset...")
        ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        texts = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            context = row.get("context", "").strip()
            response = row.get("response", "").strip()
            if not instruction or not response:
                continue
            user = f"{instruction}\n{context}".strip() if context else instruction
            text = f"User: {user}\nAssistant: {response}"
            if len(text) >= 40:
                texts.append(text)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} instruction pairs from dolly")
        return texts

    if dataset_name == "hindi_alpaca_dolly":
        logger.info("Downloading Hindi Alpaca-Dolly instruction dataset...")
        ds = load_dataset("HydraIndicLM/hindi_alpaca_dolly_67k", split="train", streaming=True)
        texts = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            output = row.get("output", "").strip()
            if not instruction or not output:
                continue
            user = f"{instruction}\n{inp}".strip() if inp else instruction
            text = f"User: {user}\nAssistant: {output}"
            if len(text) >= 30:
                texts.append(text)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} Hindi instruction pairs")
        return texts

    if dataset_name == "hinglish_alpaca_gpt4":
        logger.info("Downloading Hindi/Hinglish Alpaca-GPT4 instruction dataset...")
        ds = load_dataset("NebulaByte/alpaca-gpt4-hindi-hinglish", split="train", streaming=True)
        texts = []
        for row in ds:
            user = (row.get("input_hinglish") or row.get("input") or "").strip()
            output = (row.get("output_hinglish") or row.get("output") or "").strip()
            if not user or not output:
                continue
            text = f"User: {user}\nAssistant: {output}"
            if len(text) >= 30:
                texts.append(text)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} Hinglish instruction pairs")
        return texts

    if dataset_name == "oasst1":
        logger.info("Downloading OpenAssistant OASST1 conversations...")
        ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        pending_prompts = {}
        texts = []
        for row in ds:
            if row.get("deleted") or row.get("review_result") is False:
                continue
            msg_id = row.get("message_id", "")
            parent_id = row.get("parent_id", "")
            role = row.get("role", "")
            text = row.get("text", "").strip()
            if not msg_id or not text:
                continue
            if role == "prompter":
                pending_prompts[msg_id] = text
            elif role == "assistant" and parent_id in pending_prompts:
                pair = f"User: {pending_prompts[parent_id]}\nAssistant: {text}"
                if len(pair) >= 40:
                    texts.append(pair)
                if len(texts) >= max_samples:
                    break
        logger.info(f"Loaded {len(texts):,} OpenAssistant prompt/reply pairs")
        return texts

    if dataset_name == "openhermes":
        logger.info("Downloading OpenHermes conversations…")
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        texts = []
        for row in ds:
            convs = row.get("conversations", [])
            if not convs:
                continue
            parts = []
            for turn in convs:
                role  = turn.get("from", "")
                value = turn.get("value", "").strip()
                if role == "human":
                    parts.append(f"User: {value}")
                elif role == "gpt":
                    parts.append(f"Assistant: {value}")
            text = "\n".join(parts)
            if len(text) >= 50:
                texts.append(text)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} conversations from openhermes")
        return texts

    if dataset_name == "hindi":
        logger.info("Downloading Hindi text (IITB English-Hindi corpus)…")
        ds = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)
        texts = []
        for row in ds:
            hi = row.get("translation", {}).get("hi", "").strip()
            if len(hi) >= 20:
                texts.append(hi)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} Hindi sentences")
        return texts

    if dataset_name == "human-mix":
        import random
        logger.info("Loading human-mix (emotion + social + customer + sales + daily_dialog)…")
        texts = []
        per = max_samples // 5
        for src in ("emotion", "social_iqa", "customer_support", "sales", "daily_dialog"):
            try:
                t = _load_hf_texts(src, max_samples=per)
                texts.extend(t)
                logger.info(f"  {src}: {len(t):,}")
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"human-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "safety-mix":
        import random
        logger.info("Loading safety-mix (hate_speech + sexual_health + prosocial + empathetic)…")
        texts = []
        per = max_samples // 4
        for src in ("hate_speech", "sexual_health", "prosocial", "empathetic_dialogues"):
            try:
                t = _load_hf_texts(src, max_samples=per)
                texts.extend(t)
                logger.info(f"  {src}: {len(t):,}")
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"safety-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "spelling-mix":
        import random
        logger.info("Loading spelling-mix (spell_correction + grammar_correction + noisy_english + medit)…")
        texts = []
        per = max_samples // 4
        for src in ("spell_correction", "grammar_correction", "noisy_english", "medit"):
            try:
                t = _load_hf_texts(src, max_samples=per)
                texts.extend(t)
                logger.info(f"  {src}: {len(t):,}")
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"spelling-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "assistant-mix":
        import random
        logger.info("Loading assistant-mix (clean assistant/instruction datasets)...")
        texts = []
        sources = [
            ("oasst1",                 max_samples // 4),
            ("alpaca_gpt4",            max_samples // 4),
            ("hindi_alpaca_dolly",     max_samples // 4),
            ("hinglish_alpaca_gpt4",   max_samples // 5),
            ("dolly",                  max_samples // 10),
            ("assistant_basic",        max(200, max_samples // 50)),
        ]
        for src, n in sources:
            try:
                t = _load_hf_texts(src, max_samples=n)
                texts.extend(t)
                logger.info(f"  {src}: {len(t):,}")
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"assistant-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "clean-assistant-mix":
        return _load_hf_texts("assistant-mix", max_samples=max_samples)

    if dataset_name == "complete-mix":
        import random
        logger.info("Loading complete-mix (clean assistant + grammar + safety)...")
        texts = []
        sources = [
            ("oasst1",                 max_samples // 5),
            ("alpaca_gpt4",            max_samples // 5),
            ("hindi_alpaca_dolly",     max_samples // 5),
            ("hinglish_alpaca_gpt4",   max_samples // 5),
            ("dolly",                  max_samples // 10),
            ("wikitext2",              max_samples // 20),
            ("prosocial",              max_samples // 20),
            ("assistant_basic",        max(200, max_samples // 100)),
        ]
        for src, n in sources:
            try:
                t = _load_hf_texts(src, max_samples=n)
                texts.extend(t)
                logger.info(f"  {src}: {len(t):,}")
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"complete-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "hindi-mix":
        import random
        logger.info("Loading hindi-mix (Hindi + wikitext2 + daily_dialog)…")
        texts = []
        per = max_samples // 3
        for src in ("hindi", "wikitext2", "daily_dialog"):
            try:
                t = _load_hf_texts(src, max_samples=per)
                texts.extend(t)
                logger.info(f"  {src}: {len(t):,}")
            except Exception as e:
                logger.warning(f"  {src} failed ({e}), skipping")
        random.shuffle(texts)
        logger.info(f"hindi-mix total: {len(texts):,} texts")
        return texts

    if dataset_name == "blended_skill_talk":
        logger.info("Downloading blended_skill_talk…")
        ds = load_dataset("blended_skill_talk", split="train", streaming=True)
        texts = []
        for row in ds:
            previous = row.get("previous_utterance", [])
            chosen   = row.get("free_messages", []) or row.get("guided_messages", [])
            all_utts = (previous or []) + (chosen or [])
            convo    = "\n".join(u.strip() for u in all_utts if isinstance(u, str) and u.strip())
            if len(convo) >= 60:
                texts.append(convo)
            if len(texts) >= max_samples:
                break
        logger.info(f"Loaded {len(texts):,} conversations from blended_skill_talk")
        return texts

    # ── Standard single-field datasets ────────────────────────────────────────

    hf_name, split, field, kwargs, min_len = PRESETS[dataset_name]
    logger.info(f"Downloading '{hf_name}' (up to {max_samples:,} samples)…")
    ds = load_dataset(hf_name, split=split, streaming=True, **kwargs)

    texts = []
    for row in ds:
        t = row.get(field, "")
        if not isinstance(t, str):
            continue
        t = t.strip()
        if len(t) >= min_len:
            texts.append(t)
        if len(texts) >= max_samples:
            break

    logger.info(f"Loaded {len(texts):,} texts from '{hf_name}'")
    return texts


def _load_mixed_datasets(specs: list[str]) -> list[str]:
    """
    Load and shuffle multiple datasets together.
    Each spec is either "name" or "name:N" (N = max_samples).

    Example specs: ["tinystories:3000", "daily_dialog:2000", "wikitext2:2000"]
    """
    import random
    all_texts: list[str] = []
    for spec in specs:
        if ":" in spec:
            name, n = spec.rsplit(":", 1)
            n = int(n)
        else:
            name, n = spec, 2000
        logger.info(f"Loading {name} ({n} samples)…")
        try:
            texts = _load_hf_texts(name, max_samples=n)
            all_texts.extend(texts)
            logger.info(f"  → {len(texts):,} texts from {name}")
        except Exception as e:
            logger.warning(f"  → {name} failed: {e}, skipping")
    random.shuffle(all_texts)
    logger.info(f"Total mixed texts: {len(all_texts):,}")
    return all_texts


# ── Helper: load base or RLHF checkpoint ─────────────────────────────────────

def _load_model(prefer_rlhf: bool = True, size: str = None):
    from model.transformer import Transformer

    if size is None:
        size = _get_size()

    config  = _get_config(size)
    model   = Transformer(config)
    ckpt_rlhf = CKPT_RLHF.format(size=size)
    ckpt_base = CKPT_BASE.format(size=size)

    if prefer_rlhf and os.path.exists(ckpt_rlhf):
        ckpt = torch.load(ckpt_rlhf, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded RLHF checkpoint ({size}).")
    elif os.path.exists(ckpt_base):
        ckpt = torch.load(ckpt_base, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded base checkpoint ({size}).")
    else:
        logger.warning(f"No checkpoint found for size={size} — using random weights.")

    return model


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_info():
    from config.model_config import SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG, XLARGE_CONFIG
    from model.transformer import Transformer

    for name, cfg in [("SMALL", SMALL_CONFIG), ("MEDIUM", MEDIUM_CONFIG), ("LARGE", LARGE_CONFIG), ("XLARGE (~500M)", XLARGE_CONFIG)]:
        m = Transformer(cfg)
        print(f"\n{'-'*50}")
        print(f"  {name} CONFIG  ({m.param_count()} parameters)")
        print(f"{'-'*50}")
        print(f"  d_model    : {cfg.d_model}")
        print(f"  n_layers   : {cfg.n_layers}")
        print(f"  n_heads    : {cfg.n_heads} Q heads / {cfg.n_kv_heads} KV heads (GQA)")
        print(f"  d_ff       : {cfg.d_ff}  (SwiGLU)")
        print(f"  max_seq_len: {cfg.max_seq_len}")
        print(f"  vocab_size : {cfg.vocab_size} (cl100k_base)")


def cmd_train():
    """
    Train the base model. Saves to checkpoints/base/model.pt

    Single dataset:
      python main.py train tinystories 5000
      python main.py train daily_dialog 3000

    Multiple datasets (shuffled together — no forgetting):
      python main.py train --mix tinystories:3000 daily_dialog:2000 wikitext2:2000

    Named mixes (shortcut):
      python main.py train full-mix 8000
      python main.py train conversation-mix 5000

    Flags:
      --fresh    Start from random weights (discard existing checkpoint)
                 Without --fresh, training continues from existing checkpoint.
    """
    from model.transformer import Transformer
    from tokenizer.bpe_tokenizer import BPETokenizer
    from training.dataset import TextDataset, make_dataloader
    from training.trainer import Trainer

    size          = _get_size()
    config        = _get_config(size)
    fresh         = "--fresh" in sys.argv
    mix_mode      = "--mix"   in sys.argv
    hf_repo, hf_token = _get_hf_credentials()
    skip_flags    = {"--fresh", "--mix", "--size", "--hf-repo"}
    raw_args = [
        a for i, a in enumerate(sys.argv[2:], 2)
        if a not in skip_flags and a != size
        and not (i > 2 and sys.argv[i-1] in ("--size", "--hf-repo"))
    ]

    ckpt_path = CKPT_BASE.format(size=size)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    tokenizer = BPETokenizer()
    model     = Transformer(config)

    if fresh:
        logger.info(f"--fresh: starting from random weights ({size}).")
    elif os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Local checkpoint loaded ({size}) — continuing training.")
    elif hf_repo and hf_token:
        # Try downloading from HuggingFace Hub
        if _download_from_hub(hf_repo, hf_token, ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"HF Hub checkpoint loaded ({size}) — continuing training.")
        else:
            logger.info(f"Starting fresh ({size}).")
    else:
        logger.info(f"No checkpoint found ({size}) — starting fresh.")

    logger.info(f"Size: {size.upper()}  |  Device: {DEVICE}  |  Parameters: {model.param_count()}")

    # GPU memory clean karo — checkpoint load ke baad cached tensors hate hain
    if DEVICE == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        logger.info(f"GPU free memory after load: {free / 1e9:.2f} GB")

    # ── Load texts ────────────────────────────────────────────────────────────
    if mix_mode:
        mix_specs = [a for a in raw_args if not a.isdigit()]
        if not mix_specs:
            raise SystemExit("--mix requires dataset specs, e.g.: --mix tinystories:3000 daily_dialog:2000")
        texts = _load_mixed_datasets(mix_specs)
    elif raw_args:
        dataset_arg = raw_args[0]
        max_samples = int(raw_args[1]) if len(raw_args) > 1 else 5_000
        texts = _load_hf_texts(dataset_arg, max_samples=max_samples)
    else:
        logger.info("No dataset — using built-in sample texts.")
        texts = SAMPLE_TEXTS

    # ── Build dataset & loader ────────────────────────────────────────────────
    dataset = TextDataset(texts, tokenizer, config.max_seq_len)
    if len(dataset) == 0:
        logger.warning("Dataset chunks empty — falling back to seq_len=32.")
        cfg2    = config.model_copy(update={"max_seq_len": 32})
        dataset = TextDataset(texts, tokenizer, cfg2.max_seq_len)

    batch_size = 4 if size in ("small", "medium") else (2 if size == "large" else 1)
    epochs     = 3
    loader     = make_dataloader(dataset, batch_size=batch_size, shuffle=True)
    max_steps  = epochs * len(loader)
    warmup     = min(max(5, max_steps // 20), max_steps)

    logger.info(
        f"Dataset: {len(dataset)} chunks  |  Batch: {batch_size}  |  "
        f"Steps: {max_steps}  |  Warmup: {warmup}"
    )

    trainer = Trainer(
        model, loader,
        lr=3e-4, warmup_steps=warmup, max_steps=max_steps, device=DEVICE,
        hf_repo=hf_repo, hf_token=hf_token,
    )
    losses  = trainer.train(epochs=epochs, log_every=50, checkpoint_path=ckpt_path, save_every=500)
    trainer.save(ckpt_path)
    # Final push to HF Hub
    if hf_repo and hf_token:
        from training.trainer import _push_to_hub
        _push_to_hub(ckpt_path, hf_repo, hf_token, step=-1)
    logger.info(f"Training complete ({size}). Final loss: {losses[-1]:.4f}")


def cmd_train_rlhf():
    """
    DPO safety alignment on top of the base model.
    Saves RLHF-aligned checkpoint to checkpoints/{size}/rlhf/model.pt

    Usage:
      python main.py train-rlhf                        (uses built-in sample pairs)
      python main.py train-rlhf pairs.json             (load JSON file)
      python main.py train-rlhf --size medium          (medium model)
    """
    import json
    from tokenizer.bpe_tokenizer import BPETokenizer
    from training.rlhf_trainer import DPOTrainer

    size      = _get_size()
    ckpt_rlhf = CKPT_RLHF.format(size=size)

    # Find pairs.json arg — skip flags
    pairs_arg = None
    for a in sys.argv[2:]:
        if not a.startswith("--") and a not in (size,):
            pairs_arg = a
            break

    if pairs_arg:
        with open(pairs_arg) as f:
            pairs = json.load(f)
        logger.info(f"Loaded {len(pairs)} preference pairs from {pairs_arg}")
    else:
        pairs = SAMPLE_PREFERENCE_PAIRS
        logger.info(f"Using {len(pairs)} built-in sample preference pairs.")
        logger.info("Tip: provide a JSON file for real RLHF alignment.")

    model     = _load_model(prefer_rlhf=False, size=size)   # start from base
    tokenizer = BPETokenizer()

    os.makedirs(os.path.dirname(ckpt_rlhf), exist_ok=True)
    logger.info(f"Size: {size.upper()}  |  RLHF checkpoint → {ckpt_rlhf}")

    trainer = DPOTrainer(
        model, tokenizer, pairs,
        beta=0.1, lr=5e-5,
        warmup_steps=10, max_steps=len(pairs) * 3,
        batch_size=2, max_len=128, device=DEVICE,
    )
    losses = trainer.train(epochs=3, log_every=5, checkpoint_path=ckpt_rlhf)
    logger.info(f"RLHF training complete ({size}). Final loss: {losses[-1]:.4f}")


def cmd_train_adapter():
    """
    Train a LoRA adapter on top of the base/RLHF model.
    Saves only the adapter weights (tiny file) to checkpoints/{size}/adapters/<name>.pt

    Usage:
      python main.py train-adapter customer_service tinystories 2000
      python main.py train-adapter qa_bot wikitext2 3000 --size medium
    """
    # Collect positional args, skipping --size and its value
    size     = _get_size()
    pos_args = [a for a in sys.argv[2:] if not a.startswith("--") and a != size]

    if not pos_args:
        print("Usage: python main.py train-adapter <name> [dataset] [max_samples] [--size S]")
        sys.exit(1)

    adapter_name = pos_args[0]
    dataset_arg  = pos_args[1] if len(pos_args) > 1 else None
    max_samples  = int(pos_args[2]) if len(pos_args) > 2 else 2_000
    adapter_path = CKPT_ADAPTER.format(size=size, name=adapter_name)

    from tokenizer.bpe_tokenizer import BPETokenizer
    from training.dataset import TextDataset, make_dataloader
    from training.lora_trainer import LoRATrainer
    from model.lora_manager import LoRAManager

    os.makedirs(os.path.dirname(adapter_path), exist_ok=True)

    model     = _load_model(prefer_rlhf=True, size=size)   # build on RLHF or base
    tokenizer = BPETokenizer()
    config    = _get_config(size)

    manager = LoRAManager(rank=8, alpha=16.0)
    manager.inject(model)
    manager.freeze_base(model)

    if dataset_arg:
        texts = _load_hf_texts(dataset_arg, max_samples=max_samples)
    else:
        logger.info("No dataset specified — using built-in sample texts.")
        texts = SAMPLE_TEXTS

    dataset = TextDataset(texts, tokenizer, config.max_seq_len)
    if len(dataset) == 0:
        cfg2    = config.model_copy(update={"max_seq_len": 32})
        dataset = TextDataset(texts, tokenizer, cfg2.max_seq_len)

    epochs     = 3
    batch_size = 4 if size in ("small", "medium") else 2
    loader     = make_dataloader(dataset, batch_size=batch_size, shuffle=True)
    max_steps  = epochs * len(loader)
    warmup     = min(max(5, max_steps // 20), max_steps)

    logger.info(f"Training adapter '{adapter_name}' ({size})  |  {len(dataset)} chunks, {max_steps} steps")

    trainer = LoRATrainer(
        model, loader, manager,
        lr=1e-4, warmup_steps=warmup, max_steps=max_steps, device=DEVICE,
    )
    losses = trainer.train(epochs=epochs, log_every=50, adapter_path=adapter_path)
    logger.info(f"Adapter training complete ({size}). Final loss: {losses[-1]:.4f}")
    logger.info(f"Adapter saved → {adapter_path}")


def cmd_generate():
    from tokenizer.bpe_tokenizer import BPETokenizer
    from model.lora_manager import LoRAManager
    from inference.generator import Generator

    size         = _get_size()
    adapter_name = None
    if "--adapter" in sys.argv:
        idx = sys.argv.index("--adapter")
        if idx + 1 < len(sys.argv):
            adapter_name = sys.argv[idx + 1]

    model     = _load_model(prefer_rlhf=True, size=size)
    tokenizer = BPETokenizer()

    if adapter_name:
        adapter_path = CKPT_ADAPTER.format(size=size, name=adapter_name)
        if os.path.exists(adapter_path):
            manager = LoRAManager(rank=8, alpha=16.0)
            manager.inject(model)
            manager.load_adapter(model, adapter_path)
            logger.info(f"Loaded adapter '{adapter_name}'.")
        else:
            logger.warning(f"Adapter '{adapter_name}' not found. Using base model.")

    gen = Generator(model, tokenizer, device=DEVICE)

    prompts = [
        "User: hi\nAssistant:",
        "User: namaste\nAssistant:",
        "User: tum kya kar sakte ho\nAssistant:",
        "User: explain AI in Hinglish\nAssistant:",
        "User: I am sad\nAssistant:",
        "User: what is photosynthesis\nAssistant:",
    ]

    print("\n" + "=" * 60)
    print(f"  MODEL OUTPUT  (adapter: {adapter_name or 'none'})")
    print("=" * 60)

    for prompt in prompts:
        out = gen.generate(
            prompt,
            max_new_tokens=60,
            strategy="top_p",
            temperature=0.8,
            top_p=0.9,
        )
        # Safe print on Windows consoles that don't support Unicode
        safe_out = out.encode("ascii", errors="replace").decode("ascii")
        print(f"\nPROMPT : {prompt}")
        print(f"OUTPUT : {safe_out}")
        print("-" * 60)


def cmd_serve():
    import uvicorn
    from tokenizer.bpe_tokenizer import BPETokenizer
    from model.lora_manager import LoRAManager
    from inference.generator import Generator
    from serving.api import app, setup

    size         = _get_size()
    adapter_name = None
    if "--adapter" in sys.argv:
        idx = sys.argv.index("--adapter")
        if idx + 1 < len(sys.argv):
            adapter_name = sys.argv[idx + 1]

    model     = _load_model(prefer_rlhf=True, size=size)
    tokenizer = BPETokenizer()

    if adapter_name:
        adapter_path = CKPT_ADAPTER.format(size=size, name=adapter_name)
        if os.path.exists(adapter_path):
            manager = LoRAManager(rank=8, alpha=16.0)
            manager.inject(model)
            manager.load_adapter(model, adapter_path)
            logger.info(f"Loaded adapter '{adapter_name}' ({size}) for serving.")
        else:
            logger.warning(f"Adapter '{adapter_name}' ({size}) not found. Serving without adapter.")

    gen = Generator(model, tokenizer, device=DEVICE)
    setup(model, gen)

    logger.info(f"Model size: {size.upper()}  |  Adapter: {adapter_name or 'none'}")
    logger.info("Starting server on http://localhost:8001  |  Docs: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


def cmd_test():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(__file__),
    )
    sys.exit(result.returncode)


# ── CLI dispatch ──────────────────────────────────────────────────────────────

COMMANDS = {
    "info":          cmd_info,
    "train":         cmd_train,
    "train-rlhf":    cmd_train_rlhf,
    "train-adapter": cmd_train_adapter,
    "generate":      cmd_generate,
    "serve":         cmd_serve,
    "test":          cmd_test,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd!r}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
