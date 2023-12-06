import functools
from typing import Tuple, List

import gradio as gr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.chat_models.gigachat import GigaChat
from langchain.chat_models.openai import ChatOpenAI

import config
from dialog_history_summarizer import DialogHistorySummarizer
from history import DialogHistory
from retriever import get_vector_store, format_query, get_retriever
from strategies_interpretator import StrategyInterpreter
import nltk
from nltk.stem.snowball import SnowballStemmer

# Авторизация в сервисе GigaChat
# chat = GigaChat(credentials=config.t, verify_ssl_certs=False)
chat = ChatOpenAI(openai_api_key=config.t2)


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


# Установка NLTK и загрузка русского стеммера
nltk.download('punkt')
stemmer = SnowballStemmer("russian")


def contain_stopword(message):
    # Определение стоп-слов
    stopwords = ['оператор', 'связь', 'помощь', 'поддержка', 'хватит', 'переключи', 'позови',
                 'человек']  # Добавьте больше слов по необходимости

    words = nltk.word_tokenize(message)
    stemmed_words = [stemmer.stem(word) for word in words]

    for stopword in stopwords:
        if stemmer.stem(stopword) in stemmed_words:
            return True
    return False


def prompt_processor(chat_history: DialogHistory, message: str, stage_instruction: str,
                     dynamic_info: list) -> DialogHistory:
    dynamic_info = '\n * '.join(map(lambda x: x[0].page_content, dynamic_info))

    system_message = (f'{stage_instruction}\n Вот дополнительная информация из базы знаний, на основании которой нужно '
                      f'отвечать \n * {dynamic_info}')
    return DialogHistory(chat_history.long_chain_history, system_message)


def respond(message, chat_history, debug=False):
    global db
    """Event listener on message submit"""

    if contain_stopword(message):
        return '', chat_history + [(message, 'Сейчас свяжу с оператором \n\n Ожидаемое время ответа - 5 минут')]
    chat_history = DialogHistory([item for tpl in chat_history for item in tpl] + [message])
    db_query = DialogHistorySummarizer()(chat_history)
    print(db_query)
    documents = db(db_query)
    documents.sort(key=lambda x: x[1], reverse=True)
    print(documents)
    stage_instruction = StrategyInterpreter()(chat_history)
    print(stage_instruction)
    prompt = prompt_processor(chat_history, message, stage_instruction, documents)
    res = chat(prompt.long_chain_history)

    gradio_history = prompt.gradio_history
    gradio_history.append((message, res.content))
    # '' - to clean the input field

    return '', gradio_history, db_query, documents, stage_instruction


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(functools.partial(respond, debug=False), [msg, chatbot], [msg, chatbot])

with gr.Blocks() as debug:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    db_query = gr.Textbox(lines=5, label="db_query")
    documents = gr.Textbox(lines=7, label="documents")
    instruction = gr.Textbox(lines=5, label="instruction")
    msg.submit(functools.partial(respond, debug=True), [msg, chatbot], [msg, chatbot, db_query, documents, instruction])


class Database:
    def __init__(self):
        self.retriever = get_retriever(documents_dir='documents')

    def __call__(self, query: str):
        return self.retriever.get_relevant_documents(
            format_query(query),
        )


if __name__ == "__main__":
    db = Database()
    # demo.launch(share=True)
    debug.launch(share=True)
