import ollama
import anthropic

from src.config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODELS,
    OPENAI_API_KEY,
    MAX_TOKENS,
)
from src.retriever import retrieve

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information. Be concise and direct."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""Use the following context to answer the question.

CONTEXT:
{context}

QUESTION:
{question}"""


def ask(
    question: str,
    history: list[dict],
    provider: str = "ollama",
    model: str = None,
) -> tuple[str, list[dict]]:
    """
    Ask a question with chat history.
    Returns (answer, updated_history).
    provider: "ollama" | "anthropic" | "openai"
    model: specific model name; falls back to config defaults if None.
    """
    chunks = retrieve(question)
    user_message = build_prompt(question, chunks)
    messages = history + [{"role": "user", "content": user_message}]

    if provider == "ollama":
        resolved_model = model or OLLAMA_MODEL
        response = ollama.chat(
            model=resolved_model,
            options={"base_url": OLLAMA_BASE_URL},
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        answer = response["message"]["content"]

    elif provider == "anthropic":
        resolved_model = model or ANTHROPIC_MODELS[0]
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=resolved_model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        answer = response.content[0].text

    elif provider == "openai":
        import openai
        resolved_model = model or "gpt-4o-mini"
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=resolved_model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        answer = response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")

    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return answer, updated_history


if __name__ == "__main__":
    print("TechDocs Assistant (type 'quit' to exit)\n")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer, history = ask(question, history)
        print(f"\nAssistant: {answer}\n")