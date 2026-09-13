#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import sklearn

from src.spam_detection import spam_detection


def spy_decorator(method_to_decorate, name):
    """
    Wrap a method so calls to it are recorded on a MagicMock while the
    original implementation still runs.

    This solution to wrap a patched method without obstructing its
    implementation comes originally from
    https://stackoverflow.com/questions/25608107/
    """
    mock = MagicMock(name="%s method" % name)

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)
    wrapper.mock = mock
    return wrapper


class TestSpamDetection(unittest.TestCase):

    def test_first(self):
        accuracy, total, misclassified = spam_detection(
            random_state=0, fraction=0.1)
        self.assertEqual(
            accuracy, 0.96,
            msg="Incorrect accuracy, when random_state=0 and fraction=0.1!")
        self.assertEqual(
            total, 75,
            msg="Incorrect sample size, when random_state=0 and "
                "fraction=0.1!")
        self.assertEqual(
            misclassified, 3,
            msg="Incorrect misclassified count, when random_state=0 and "
                "fraction=0.1!")

    def test_second(self):
        accuracy, total, misclassified = spam_detection(
            random_state=5, fraction=0.1)
        self.assertAlmostEqual(
            accuracy, 0.9066666666666666,
            msg="Incorrect accuracy, when random_state=5 and fraction=0.1!")
        self.assertEqual(
            total, 75,
            msg="Incorrect sample size, when random_state=5 and "
                "fraction=0.1!")
        self.assertEqual(
            misclassified, 7,
            msg="Incorrect misclassified count, when random_state=5 and "
                "fraction=0.1!")

    def test_calls(self):
        predict_method = spy_decorator(
            sklearn.naive_bayes.MultinomialNB.predict, "predict")
        score_method = spy_decorator(
            sklearn.naive_bayes.MultinomialNB.score, "score")
        fit_method = spy_decorator(
            sklearn.naive_bayes.MultinomialNB.fit, "fit")
        with patch("src.spam_detection.train_test_split",
                   wraps=sklearn.model_selection.train_test_split) as tts, \
             patch("src.spam_detection.accuracy_score",
                   wraps=sklearn.metrics.accuracy_score) as acs, \
             patch.object(sklearn.naive_bayes.MultinomialNB, "fit",
                          new=fit_method), \
             patch.object(sklearn.naive_bayes.MultinomialNB, "predict",
                          new=predict_method), \
             patch.object(sklearn.naive_bayes.MultinomialNB, "score",
                          new=score_method), \
             patch("src.spam_detection.MultinomialNB",
                   wraps=sklearn.naive_bayes.MultinomialNB) as mnb:

            random_state = 7
            accuracy, total, misclassified = spam_detection(
                random_state, fraction=0.1)

            # Check that train_test_split is called with correct parameters
            tts.assert_called_once()
            args, kwargs = tts.call_args
            self.assertIn(
                "random_state", kwargs,
                msg="You did not specify the random_state argument to "
                    "train_test_split!")
            self.assertEqual(
                kwargs["random_state"], random_state,
                msg="Incorrect random_state argument to train_test_split!")
            if "train_size" in kwargs:
                self.assertEqual(
                    kwargs["train_size"], 0.75,
                    msg="Incorrect train_size argument to "
                        "train_test_split!")
            if "test_size" in kwargs:
                self.assertEqual(
                    kwargs["test_size"], 0.25,
                    msg="Incorrect test_size argument to train_test_split!")
            # Check that accuracy_score is called
            self.assertTrue(
                acs.call_count == 1 or score_method.mock.call_count == 1,
                msg="Expected that either the accuracy_score function or "
                    "the score method is called exactly once!")

            # Check that MultinomialNB is called
            mnb.assert_called_once()

            # Check that fit and predict methods of MultinomialNB object
            # are called
            predict_method.mock.assert_called()
            fit_method.mock.assert_called()


if __name__ == '__main__':
    unittest.main()
