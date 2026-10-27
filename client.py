from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
model = "claude-haiku-4-5"


def add_user_message(messages, text):
    """Append a user turn with the given text to the messages list, in place."""
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    """Append an assistant turn to the messages list, in place.

    As the last message it acts as a prefill: the model continues from this text.
    """
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages, system=None, stop_sequences: list[str] | None = None):
    """Send the messages to the model and return the text of its reply.

    Args:
        messages: Conversation so far, built with the add_*_message helpers.
        system: Optional system prompt.
        stop_sequences: Optional strings that end generation when produced.
            The stop string itself is not included in the returned text.
    """
    params = {
        "model": model,
        "max_tokens": 4000,
        "messages": messages,
    }

    if system:
        params["system"] = system

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)
    return message.content[0].text