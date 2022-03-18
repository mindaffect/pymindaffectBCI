"""This module uses an n-gram frequency list for word autocompletion and prediction.

The module contains a single class ``WordPrediction``.
Make an instance of this class in order to use it.
"""

#  Copyright (c) 2021,
#  Authors: Thomas de Lange, Thomas Jurriaans, Damy Hillen, Joost Vossers, Jort Gutter, Florian Handke, Stijn Boosman
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os

from mindaffectBCI.examples.presentation.smart_keyboard.settings_manager import SettingsManager


class WordPrediction:
    """This class autocompletes and predicts words"""

    __instance = None

    @staticmethod
    def get_instance(update_ngram_depth=None):
        """ Static access method. """
        if WordPrediction.__instance is None:
            WordPrediction(update_ngram_depth)
        return WordPrediction.__instance

    def __init__(self, update_ngram_depth=None):
        """
                   Args:
                       update_ngram_depth (int): The depth to which the ngram looks
               """
        if WordPrediction.__instance is not None:
            raise Exception("An instance of this class already exists")
        else:
            WordPrediction.__instance = self

            self.settings_manager = SettingsManager.get_instance()
            self.settings_manager.attach(self)  # attach the object so it gets observed when the language changes

            self.language = self.settings_manager.get_language()

            # Make sure there is a place to store the n-gram files
            self.basedirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"dictionaries")
            self.ensure_directory(self.basedirectory)
            self.ensure_directory(os.path.join(self.basedirectory,'ngram_files'))
            self.ensure_directory(os.path.join(self.basedirectory,'ngram_files','pretrained'))
            self.ensure_directory(os.path.join(os.path.join(self.basedirectory,'ngram_files','user_trained')))

            # Load the n-gram frequency list for the specified language
            self.pretrained_path = os.path.join(self.basedirectory, "ngram_files", "pretrained",
                                                self.language + ".ngrams.json")
            self.user_trained_path = os.path.join(self.basedirectory, "ngram_files", "user_trained",
                                                  self.language + ".ngrams.json")
            self.pretrained = self.load_ngrams(self.pretrained_path)
            self.user_trained = self.load_ngrams(self.user_trained_path)

            self.update_ngram_depth = update_ngram_depth

    @staticmethod
    def load_ngrams(path):
        """Loads the n-gram dictionary from the specified file. If it doesn't exist, returns a new n-gram dictionary.

        Args:
            path (str): The file path of the n-gram dictionary to load.

        Return:
             The n-gram dictionary.
        """
        if isinstance(path,str):
            if not os.path.exists(path):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)),path)
        if os.path.isfile(path):  # Load existing n-gram dictionary
            return json.load(open(path, "r"))
        return {"__max-depth__": 1}  # Create new n-gram dictionary to count frequencies from scratch

    def update(self):
        """Updates the language at use if it has been changed."""
        new_lang = self.settings_manager.get_language()
        if self.language == new_lang:
            return

        self.language = new_lang

        self.pretrained_path = os.path.join(self.basedirectory, "ngram_files", "pretrained", self.language + ".ngrams.json")
        self.user_trained_path = os.path.join(self.basedirectory, "ngram_files", "user_trained",
                                              self.language + ".ngrams.json")
        self.pretrained = self.load_ngrams(self.pretrained_path)
        self.user_trained = self.load_ngrams(self.user_trained_path)

    def update_frequencies(self, text):
        """Updates the frequencies of n-grams in the (user trained) n-grams file of the specified language.

        Args:
            text: The text to update the n-gram frequency file with
        """

        words = text.split(" ")
        while "" in words:
            words.remove("")

        for offset in range(self.update_ngram_depth):  # Different n-gram sizes
            for start in range(len(words) - offset):  # Different starting words
                ngram = words[start:start + offset + 1]
                front = " ".join(ngram[:-1])
                back = ngram[-1]
                if front not in self.user_trained:  # Check if the front part of the n-gram exists in the dictionary
                    self.user_trained[front] = {"__counter__": 0}
                if back in self.user_trained[front]:  # Increase counter if the n-gram has been seen before
                    self.user_trained[front][back] += 1
                else:
                    self.user_trained[front][back] = 1  # Set initial count if the n-gram is new
                self.user_trained[front]["__counter__"] += 1

        self.user_trained["__max-depth__"] = max(self.user_trained["__max-depth__"], self.update_ngram_depth)

        json.dump(self.user_trained, open(os.path.join(self.basedirectory, "ngram_files", "user_trained",
                                                       self.language + ".ngrams.json"), "w"))  # Update the file

    def predict(self, text="", n=3, depth=3, all=False):
        """Gives n predictions based on the context and the language's n-gram frequency file.

        Args:
            text (str): The context to base the predictions on. (default is "")
            n (int): The amount of predictions to return. (default is 3)
            depth (int): The maximal n-gram size used for prediction. (higher = more context used) (default is 3)
            all (bool): If true, instead of returning the best n predictions, it returns all predictions. (default is False)

        Returns:
            A list of n predictions based on the context and n-gram frequency file
        """
        front_ls = text.split(" ")
        while "" in front_ls:
            front_ls.remove("")

        # If for some reason no dictionary is loaded, send empty predictions
        if not self.user_trained and not self.pretrained:
            return [""] * n

        fronts = [""] * depth
        result = []
        for d in range(1, depth):  # Generate the front parts of the n-grams
            fronts[d] = " ".join(front_ls[-d:])

        for front in fronts:  # Get the n-gram predictions for all different length fronts
            result += self.get_predictions(front)

        result.sort(reverse=True)  # Sort predictions based on frequency (highest first)

        result = [word for freq, word in result]  # Take the words away from the frequencies

        self.remove_first_duplicates(result, n)  # Make sure all predictions are unique.

        if all:
            return result

        return (result + [""] * n)[:n]  # Return the predictions and make sure the list is exactly n elements long

    def get_predictions(self, front):
        """Retrieves the predictions and their frequencies based on the frequency file.

        Args:
            front (list(str)): The first n-1 words from the n-gram that are used for prediction context

        Returns:
            A list of tuples (frequency, word) for the predictions based on the n-gram front
        """
        entries = []
        if self.user_trained and front in self.user_trained:  # Check if we can take ngrams from user_trained
            entries.extend(self.user_trained[front])
        if self.pretrained and front in self.pretrained:      # Check if we can take ngrams from pretrained
            entries.extend(self.pretrained[front])
        ls = [(self.get_probability(front, back), back) for back in entries if back != "__counter__"]
        return ls

    def get_probability(self, front, back):
        """Calculates the normalized frequency of the prediction ``back`` given context ``front``.

        Args:
            front (str): The n-gram front which specifies the context
            back (str): The word that is predicted based on the specified front

        Returns:
            The probability of ``back`` occurring given the context ``front``
        """
        user_trained = (0, 0)
        if front in self.user_trained:
            user_trained = (user_trained[0], self.user_trained[front]["__counter__"])
            if back in self.user_trained[front]:
                user_trained = (self.user_trained[front][back], user_trained[1])

        pretrained = (0, 0)
        if front in self.pretrained:
            pretrained = (pretrained[0], self.pretrained[front]["__counter__"])
            if back in self.pretrained[front]:
                pretrained = (self.pretrained[front][back], pretrained[1])

        p_back = 0 if (user_trained[1] + pretrained[1]) == 0 else ((user_trained[0] + pretrained[0]) / (user_trained[1] + pretrained[1]))
        return p_back

    def remove_first_duplicates(self, ls, n):
        """Removes the duplicates from the first n elements in the list so it can be used as prediction suggestions.

        Args:
            ls (list(str)): The list of predictions
            n (int): Indicates that the first n predictions have to be unique

        Returns:
            The (partially) filtered list
        """
        unstable = True  # Tracks whether the first n items are 'stable' (so the first part of the list doesn't change)
        i = 0
        while len(ls) > 1 and unstable:
            unstable = False
            j = i + 1  # Start looking for duplicates after the specified word at index i
            while j < len(ls) and j < n:
                while j < len(ls) and j < n and ls[i] == ls[j]:  # As long as ls[i] and ls[j] are the same, remove ls[j]
                    ls.pop(j)
                    unstable = True  # If a duplicate is removed, the first part of the list is not stable yet
                j += 1  # As soon as ls[i] and ls[j] are different, move j to the next index
            i = (i + 1) % n  # Go to the next i (keep looping through the first n elements)
        return ls

    def autocomplete(self, text="", n=3):
        """Autocomplete words based on the specified language's n-gram frequency file.

        Args:
            text (str): The part of the word that has already been typed.
            n (int): How many suggestions to return.

        Returns:
            A list of n words which are the autocompletion suggestions for the word.
        """

        last = text.split(" ")[-1]
        front_txt = " ".join(text.split(" ")[:-1])
        # Retrieve all matching words and put them in tuples of (frequency, word)
        matches = [pred for pred in self.predict(front_txt, n=1000, all=True) if
                   last.lower() == pred[:len(last)].lower()] + [""] * n
        matches = self.copy_capitalization(last, matches[:n])
        matches = self.remove_first_duplicates(matches, n)
        return (matches + [""]*n)[:n]

    def copy_capitalization(self, word, words):
        """Copies capitalization from 'word' to the words in list 'words'.

        Args:
            word (str): The word to copy the capitalization from
            words (list): The list of words to copy the capitalization to

        Returns:
            The list
        """
        upper_cases = [index for index, c in enumerate(word) if c.lower() != c]  # Determine capitalized letters.
        preds = [''.join([c.upper() if index in upper_cases else c for index, c in enumerate(word)]) for word in
                 words]  # Set capital letters in suggestion_keys
        return preds

    @staticmethod
    def ensure_directory(path):
        """Checks whether the specified directory exists. If not, it creates it.

        Args:
            path (str): The directory path to be checked.
        """
        if not os.path.isdir(path):
            print(os.path.join("Directory \'.", path) + " does not exist yet. Creating it now.")
            os.mkdir(path)
