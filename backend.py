from typing import Tuple, List

import gradio as gr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.chat_models.gigachat import GigaChat

import config
from dialog_history_summarizer import DialogHistorySummarizer
from history import DialogHistory
from retriever import get_vector_store, format_query, get_retriever
from strategies_interpretator import StrategyInterpreter

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


def get_sell_stage_instruction(message, chat_history) -> str:
    return 'Ты ии-ассистент'


def prompt_processor(chat_history: DialogHistory, message: str, stage_instruction: str,
                     dynamic_info: list) -> DialogHistory:
    dynamic_info = '\n * '.join(map(lambda x: x.page_content,  dynamic_info))

    system_message = (f'{stage_instruction}\n Вот дополнительная информация из базы знаний, на основании которой нужно '
                      f'отвечать \n * {dynamic_info}')
    return DialogHistory(chat_history.long_chain_history, system_message)


def respond(message, chat_history):
    global db
    """Event listener on message submit"""
    chat_history = DialogHistory([item for tpl in chat_history for item in tpl] + [message])
    db_query = DialogHistorySummarizer()(chat_history)
    documents = db(db_query)
    stage_instruction = StrategyInterpreter()(chat_history)

    prompt = prompt_processor(chat_history, message, stage_instruction, documents)

    res = chat(prompt.long_chain_history)

    gradio_history = prompt.gradio_history
    gradio_history.append((message, res.content))
    # '' - to clean the input field
    return '', gradio_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    sell_stage = gr.State()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])


class Database:
    def __init__(self):
        self.retriever = get_retriever(documents_dir='documents')

    def __call__(self, query: str):
        return self.retriever.get_relevant_documents(
            format_query(query),
        )


if __name__ == "__main__":
    db = Database()
    demo.launch(share=True)
