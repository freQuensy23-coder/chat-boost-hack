from langchain.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json

import config
from history import DialogHistory
from utils import batched
from instructions import instructions, stages

token = 'YjNjOGIwMjctZGY2Ny00MTk0LWFmZDUtZGM2NjBjOTYwNjIwOmI3MTRjMmJjLTIzMjEtNDc3OC1iNGZjLTgwNGVmYTIxODQxZg=='
class StrategyInterpreter:
    """Summarize dialog history into one question.
        input:
            User: У вас есть годовая подписка
            AI: Да, она стоит 1999 в год и дает много бонусов, рассказать подробнее?
            User: Да, давай
        output:
            <Иструкция по этапу продажи из документа>
    """

    # Singleton init
    def __init__(self):
        self.llm = GigaChat(credentials=token, verify_ssl_certs=False)
        file_path = 'SalesStepsDescriptionAndActions.txt'

        with open(file_path, 'r') as file:
            self.strategies = file.read()
        self.stages = stages
        self.instructions = instructions

    def __call__(self, history: DialogHistory) -> str:
        system_message = "Тебе на вход поступил диалог клиента и менеджера по продажам в чате, а также документ с описаниями шагов продаж. Определи, какой это шаг продаж и инструкцию к этому шагу, описанную в документе"
        query = f"История диалога:  \n\n f{history.__str__()} \n\n Описания шагов продаж : \n\n {self.strategies}"
        prompt = [SystemMessage(content=system_message), HumanMessage(content=query)]
        answer = self.llm(prompt).content

        for (stage_id, stage) in enumerate(stages, 1):
            if answer.count(stage) > 0:
                return self.instructions[stage_id]
        return self.instructions[3]


if __name__ == '__main__':
    interpreter = StrategyInterpreter()
    while True:
        input_text = input('Ввод в формате фразы через \\n')
        history = input_text.split('\n')

        history = DialogHistory(*batched(history, 2), system_message=None)
        answer = interpreter(history)
        print(answer)