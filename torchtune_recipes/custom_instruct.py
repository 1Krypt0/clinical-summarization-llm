from torchtune.data import InstructTemplate


class DefaultInstruct(InstructTemplate):

    @classmethod
    def format(cls, sample, column_map=None) -> str:

        return sample["text"]
