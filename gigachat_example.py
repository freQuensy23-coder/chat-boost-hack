from typing import Tuple, List

import gradio as gr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models.gigachat import GigaChat

import config

# Авторизация в сервисе GigaChat
chat = GigaChat(credentials=config.token, verify_ssl_certs=False)


def create_gigachat_history(history: [Tuple[str, str]],
                            message: str,
                            base_prompt: str) -> List:
    """Create a list of Long chain message objects based on a history of messages in gradio format"""
    base_prompt = [
        SystemMessage(
            content=base_prompt
        )
    ]
    result = []
    for user_message, bot_message in history:
        result.append(HumanMessage(content=user_message))
        result.append(AIMessage(content=bot_message))
    if message:
        result.append(HumanMessage(content=message))
    return base_prompt + result


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    system_prompt = gr.Textbox(lines=5, label="System prompt", value="Ты ии помощник")
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chat_history, system_prompt_str):
        gigachat_messages = create_gigachat_history(chat_history, message, system_prompt_str)
        res = chat(gigachat_messages)
        chat_history.append((message, res.content))
        # '' - to clean the input field
        return '', chat_history


    msg.submit(respond, [msg, chatbot, system_prompt], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
