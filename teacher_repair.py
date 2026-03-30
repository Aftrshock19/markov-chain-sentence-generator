#!/usr/bin/env python3
from typing import Optional

from teacher_llm import TeacherLLM, TeacherConfig



def repair_sentence_with_teacher(
    sentence: str,
    lemma: str,
    rank: int,
    pos: str,
    translation: Optional[str] = None,
    config: Optional[TeacherConfig] = None,
) -> str:
    teacher = TeacherLLM(config=config)
    messages = [
        {
            "role": "system",
            "content": "You rewrite Spanish learner example sentences. Return only one sentence and nothing else.",
        },
        {
            "role": "user",
            "content": (
                "Rewrite this sentence so it is short, natural, learner friendly, and still uses Spain Spanish.\n"
                f"Keep the exact target word unchanged: {lemma}\n"
                f"Part of speech: {pos}\n"
                f"Rank: {rank}\n"
                f"Gloss: {translation or ''}\n"
                f"Sentence: {sentence}\n"
                "Return one sentence only."
            ),
        },
    ]
    try:
        response = teacher.client.chat.completions.create(
            model=teacher.config.model,
            messages=messages,
            temperature=min(teacher.config.temperature, 0.25),
            max_tokens=min(teacher.config.max_tokens, 80),
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return ""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return lines[0] if lines else ""
