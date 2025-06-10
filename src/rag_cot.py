import os
import dspy

from src.retriever import LocalRetriever
from src.config import API_KEY

# Set OpenAI key and select GPT-4.1 full
llm = dspy.LM('openai/gpt-4.1-2025-04-14', api_key=API_KEY)


class AnswerSig(dspy.Signature):
    context  = dspy.InputField()
    question = dspy.InputField()
    answer   = dspy.OutputField()


class CoTSig(dspy.Signature):
    context   = dspy.InputField()
    question  = dspy.InputField()
    rationale = dspy.OutputField()


class RAGCoT(dspy.Module):
    def __init__(self, retriever: LocalRetriever, k: int = 4):
        super().__init__()
        self.retriever = retriever
        self.cot       = dspy.Predict(llm, CoTSig)
        self.final     = dspy.Predict(llm, AnswerSig)
        self.k = k

    def forward(self, question: str) -> str:
        chunks  = self.retriever(question, k=self.k)
        context = "\n\n".join(chunks)
        chain   = self.cot(context=context, question=question).rationale
        ans     = self.final(
            context=f"{context}\n\nReasoning:\n{chain}",
            question=question
        ).answer
        return ans