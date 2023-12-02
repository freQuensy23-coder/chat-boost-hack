from langchain.chat_models import GigaChat

import config


class DialogHistorySummarizer:
    """Summarize dialog history into one question.
        input:
            User: У вас есть годовая подписка
            AI: Да, она стоит 1999 в год и дает много бонусов, рассказать подробнее?
            User: Да, давай
        output:
            Расскажите мне подробнее о вашей годовой подписке за 1999 в год и ее бонусах.
    """

    # Singleton init
    def __init__(self):
        self.llm = GigaChat(credentials=config.token, verify_ssl_certs=False)

    def __call__(self, ):
