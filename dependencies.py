class Retriever(object):
    def get_relevant_documents(self, query: str) -> list[str]:
        return ['One relevant document', 'Another relevant document']


class LLM(object):
    prompt_template = 'Never forget your name is {salesperson_name}.\n{relevant_documents}\n{dialogue}'
    prompt_arguments = {'salesperson_name': 'Oleg'}

    def __call__(
        self,
        dialogue: list[str],
        relevant_documents: list[str],
    ) -> str:
        _ = self.prompt_template.format(
            **self.prompt_arguments,
            dialogue=dialogue,
            relevant_documents=relevant_documents,
        )
        return '42'


class SuccessAnalyzer(object):
    def __call__(self, dialogue: list[str]) -> None:
        # check if there is name + phone number in dialogue
        # logging + metrics
        ...
