from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage


class DialogHistory:
    def __init__(self, history, system_message: str | None = None):
        """:param history: in any format - gradio or long chain
                [(user_message, bot_message), ...]
                or
                [HumanMessage(content=user_message), AIMessage(content=bot_message), ...]
            :param system_message: system message to start the dialog with
                """
        self._history: List[BaseMessage] = []
        if isinstance(history[0], tuple):
            for user_message, ai_message in history:
                self._history.append(HumanMessage(content=user_message))
                self._history.append(AIMessage(content=ai_message))

        elif isinstance(history[0], BaseMessage):
            if isinstance(history[0], HumanMessage) or isinstance(history[0], SystemMessage):
                self._history = history
            if isinstance(history[0], AIMessage):
                raise ValueError("First message in history can't be AIMessage")

        elif isinstance(history[0], str):
            message_source = 'user'
            for message in history:
                if message_source == 'user':
                    self._history.append(HumanMessage(content=message))
                    message_source = 'ai'
                else:
                    self._history.append(AIMessage(content=message))
                    message_source = 'user'

        self._system_message = system_message
        if not isinstance(self._history[0], SystemMessage) and system_message:
            self._history = [SystemMessage(content=system_message)] + self._history

    @property
    def gradio_history(self) -> List[tuple]:
        return [(user_message.content, ai_message.content) for user_message, ai_message in
                zip(self._history[::2], self._history[1::2])]

    @property
    def long_chain_history(self):
        return self._history

    @property
    def system_message(self):
        return self._system_message

    def add_message(self, user_message: str, bot_message: str | None = None):
        self._history.append(HumanMessage(content=user_message))
        if bot_message:
            self._history.append(AIMessage(content=bot_message))

    def __str__(self):
        """Return User: message \n AI: message \n ..."""
        source = 'user'
        res = ''
        for message in self._history:
            if source == 'user':
                res += f'User: {message.content}\n'
                source = 'ai'
            else:
                res += f'AI: {message.content}\n'
                source = 'user'

        return res.strip()


if __name__ == '__main__':
    history = DialogHistory([('Привет', 'Привет'), ('Как дела?', 'Нормально'), ('Чем занимаешься?', 'Пишу код')],
                            system_message='Ты ии Друг')
    history.add_message('Как дела?')
    print(history.gradio_history)
    print(history.long_chain_history)
    history.add_message('Чем занимаешься?')
    print(history.gradio_history)
    print(history.long_chain_history)
    history.add_message('Чем занимаешься?', 'Пишу код')
    print(history.gradio_history)
    print(history.long_chain_history)

    history2 = DialogHistory(['Привет'])
    print(history2.gradio_history)
    print(history2.long_chain_history)
    print(str(history2))
