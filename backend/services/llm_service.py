"""
LLM Service

- ส่ง Context เข้า Local LLM
- รับคำตอบกลับมา
"""

import ollama


def ask_llm(
    question,
    context
):

    prompt = f"""
Answer using only the provided context.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]