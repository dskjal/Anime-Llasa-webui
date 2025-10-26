# https://huggingface.co/spaces/OmniAICreator/Anime-Llasa-3B-Captions-Demo/blob/main/app.py

import re

REPLACE_MAP: dict[str, str] = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",  # dashes
    r"[\uff5e\u301C]": "ー",  # wave dash variants
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}

FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {chr(full): chr(half) for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))}
)

INVALID_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    r"\u0041-\u005A\u0061-\u007A"
    r"\u0030-\u0039"
    r"。、「」、!?…♪♡○（）]"  # allow （）
)

def normalize_caption(text: str) -> str:
    """
    Normalize text to match the preprocessing rules.
    """
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)
    text = re.sub(r"…{3,}", "……", text)
    return text

def build_system_text(meta: dict) -> str:
    """
    Build system text exactly like preprocessing (fixed order/keys).
    """
    def v(key: str) -> str:
        val = meta.get(key)
        return val if val else ""
    return (
        f"emotion: {v('emotion')}\n"
        f"profile: {v('profile')}\n"
        f"mood: {v('mood')}\n"
        f"speed: {v('speed')}\n"
        f"prosody: {v('prosody')}\n"
        f"pitch_timbre: {v('pitch_timbre')}\n"
        f"style: {v('style')}\n"
        f"notes: {v('notes')}\n"
        f"caption: {v('caption')}"
    )