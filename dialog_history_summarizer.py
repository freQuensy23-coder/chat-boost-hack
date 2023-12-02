from langchain.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import config
from history import DialogHistory
from utils import batched


class DialogHistorySummarizer:
    """Summarize dialog history into one question.
        input:
            User: У вас есть годовая подписка
            AI: Да, она стоит 1999 в год и дает много бонусов, рассказать подробнее?
            User: Да, давай
        output:
            Расскажите мне подробнее о вашей годовой подписке за 1999 в год и ее бонусах.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DialogHistorySummarizer, cls).__new__(cls)
            # Инициализация объекта
            cls._instance.llm = GigaChat(credentials=config.token, verify_ssl_certs=False)
        return cls._instance

    def __init__(self):
        self.llm = GigaChat(credentials=config.token, verify_ssl_certs=False)

    def __call__(self, history: DialogHistory) -> str:
        system_message = """Тебе дан диалог между пользователем и менеджером. Переформулируй последнее пользователя сообщение в один понятный вопрос, учитывая контекст диалога"""
        prompt = [SystemMessage(content=system_message), HumanMessage(
            content=str(history[:-1]) + f'\n\n Последний вопрос от пользователя: {history[-1].content}')]
        res = self.llm(prompt)
        return res.content.replace('\n', '').replace('?', '')


if __name__ == '__main__':
    while True:
        input_text = input('Ввод в формате фразы через \\n')
        history = input_text.split('\n')

        history = DialogHistory(*batched(history, 2), system_message=None)
        summarizer = DialogHistorySummarizer()
        print(summarizer(history))
