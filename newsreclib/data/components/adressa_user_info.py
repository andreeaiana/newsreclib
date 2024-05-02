# Adapted from https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py

from typing import List, Optional

import numpy as np


class UserInfo:
    """
    Attributes:
        train_date_split:
            A string with the date before which click behaviors are included in the history of a user.
        test_date_split:
            A string with the date after which click behaviors are included in the test set.

    """

    def __init__(
        self,
        train_date_split: int,
        test_date_split: int,
    ) -> None:
        self.hist_news = []
        self.hist_time = []

        self.train_news = []
        self.train_time = []

        self.test_news = []
        self.test_time = []

        self.train_date_split = train_date_split
        self.test_date_split = test_date_split

    def update(self, nindex: int, click_time: int, date: str):
        """
        Args:
            nindex:
                The index of a news article.
            click_time:
                The time when the user clicked on the news article.
            date:
                The processed click time used to assign the sample into the `history` of the user, the `train` or the `test` set.

        """
        if date >= self.train_date_split and date < self.test_date_split:
            self.train_news.append(nindex)
            self.train_time.append(click_time)
        elif date >= self.test_date_split:
            self.test_news.append(nindex)
            self.test_time.append(click_time)
        else:
            self.hist_news.append(nindex)
            self.hist_time.append(click_time)

    def sort_click(self):
        """Sorts user clicks by time in ascending order."""
        self.train_news = np.array(self.train_news)
        self.train_time = np.array(self.train_time, dtype="int32")

        self.test_news = np.array(self.test_news)
        self.test_time = np.array(self.test_time, dtype="int32")

        self.hist_news = np.array(self.hist_news)
        self.hist_time = np.array(self.hist_time, dtype="int32")

        order = np.argsort(self.train_time)
        self.train_news = self.train_news[order]
        self.train_time = self.train_time[order]

        order = np.argsort(self.test_time)
        self.test_news = self.test_news[order]
        self.test_time = self.test_time[order]

        order = np.argsort(self.hist_time)
        self.hist_news = self.hist_news[order]
        self.hist_time = self.hist_time[order]
