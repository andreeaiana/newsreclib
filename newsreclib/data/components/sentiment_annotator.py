from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VADERSentimentAnnotator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = SentimentIntensityAnalyzer()

    def forward(self, text: str) -> Tuple[str, float]:
        """Computes the sentiment orientation of a text.

        Args:
            text:
                A piece of text.

        Returns:
            A tuple containing the sentiment class and score of the text.
        """
        # sentiment polarity score
        sent_score = self.model.polarity_scores(text)["compound"]

        # sentiment class
        if sent_score >= 0.05:
            sent_class = "positive"
        elif sent_score <= -0.05:
            sent_class = "negative"
        else:
            sent_class = "neutral"

        return (sent_class, sent_score)


class BERTSentimentAnnotator(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer_max_len: int,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=tokenizer_max_len
        )
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, text: str) -> Tuple[str, float]:
        """Computes the sentiment orientation of a text.

        Args:
            text:
                A piece of text.

        Returns:
            A tuple containing the sentiment class and score of the text.
        """
        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        logits = self.model(**encoded_input).logits
        scores = F.softmax(logits[0], dim=0).detach().numpy()

        sent_class = self.config.id2label[scores.argmax()]
        if sent_class == "positive":
            sent_score = scores[scores.argmax()]
        elif sent_class == "negative":
            sent_score = -scores[scores.argmax()]
        else:
            sent_score = (1 - scores[scores.argmax()]) * (scores[-1] - scores[0])

        return (sent_class, sent_score)
