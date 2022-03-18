"""
This script allows users to generate n-gram frequency files, which are used for word prediction, based on input .txt files.
If the output file already exists, it is loaded and updated with the specified files.

**Usage**::

        python create_ngram_file.py [-flags] [args]

The following flags can be used:
 * ``dictionary``: uses the unigrams to output a word frequency dictionary used in word correction to the directory dictionaries/
 * ``punctuation``: filters out (most) punctuation based on some rules specified in the script before counting n-grams.
 * ``capitalization``: changes the input text to be completely lower case before counting n-grams.

The following parameters have to be specified:
 * ``language_code``: A 2-letter code representing the language that the n-gram file is generated for.
 * ``max_ngram_size``: Larger n-grams means more context used for prediction. A value of 3 means all uni-, bi- and trigrams are counted.
 * ``input_path``: A path to a .txt file or a folder containing .txt files. In case a folder is specified, the script counts the n-grams for all .txt files in the folder.
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

import re as regex
import argparse
import string
import json
import sys
import os


class CreateNgramFile:
    @staticmethod
    def generate():
        """
        This is the only function in this script. It generates the n-gram frequency file according to the specified commandline arguments.
        """
        # Parsing the commandline input
        parser = argparse.ArgumentParser()
        parser.add_argument("language_code", help="A two-letter code representing the language of the resulting n-gram frequency file.")
        parser.add_argument("max_ngram_size", help="The maximum size of the n-grams counted. A max size of 3 means that all uni-, bi-, and trigrams are counted.")
        parser.add_argument("input_path", help="The path of the input file or directory containing multiple input files.")
        parser.add_argument("-dictionary", action="store_true", help="Create a frequency dictionary for word correction.")
        parser.add_argument("-punctuation", action="store_true", help="Filter out the punctuation before counting the n-grams.")
        parser.add_argument("-capitalization", action="store_true", help="Filter out the capitalization before counting the n-grams.")
        args = parser.parse_args()

        # Make sure there is a place to store the n-gram files
        basedirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"dictionaries")
        CreateNgramFile.ensure_directory(basedirectory)
        CreateNgramFile.ensure_directory(os.path.join(basedirectory,'ngram_files'))
        CreateNgramFile.ensure_directory(os.path.join(basedirectory,'ngram_files','pretrained'))
        CreateNgramFile.ensure_directory(os.path.join(os.path.join(basedirectory,'ngram_files','user_trained')))

        # Checking the given arguments' validity
        try:
            N = int(args.max_ngram_size)
            if N < 1:
                print("Please enter a max n-gram size > 0!", file=sys.stderr)
                exit()
        except:
            print("Please enter a valid number for the n-gram size!", file=sys.stderr)
            exit()

        INPUT_PATH = args.input_path  # Either a .txt file or a directory with .txt files
        LANGUAGE = args.language_code.upper()
        if len(LANGUAGE) > 2:
            print("Please give a valid language code!", file=sys.stderr)
            exit()

        files = []
        if os.path.isfile(INPUT_PATH):  # User selected a single file
            files = [INPUT_PATH]

        if os.path.isdir(INPUT_PATH):  # User selected a folder
            files = [INPUT_PATH + file for file in os.listdir(INPUT_PATH) if file[-4:] == ".txt"]

        if len(files) == 0:  # No usable file(s) found
            print("No input file(s) found!", file=sys.stderr)
            exit()

        # Set output file name:
        OUTPUT_NAME = os.path.join(basedirectory, "ngram_files", "pretrained", LANGUAGE + ".ngrams.json")

        if not os.path.isfile(OUTPUT_NAME):  # Output file does not exist yet
            print("N-gram frequency list does not exist yet. Creating new file \'" + OUTPUT_NAME + "\'.")
            ngrams = dict()
        else:  # Output file already exists
            print("Existing n-gram frequency list found. Using file \'" + OUTPUT_NAME + "\'.")
            ngrams = json.load(open(OUTPUT_NAME, "r"))

        def update_dictionary(front, word):
            """Updates the n-gram frequencies according to the parameters

            Args:
                front (str): the n-1 preceding words of the n-gram
                word (str): the last word of the n-gram
            """
            if front in ngrams:  # first n-1 words of n-gram exist in the dictionary
                if word in ngrams[front]:  # n-gram already exists
                    ngrams[front][word] += 1
                else:  # n-gram does not exist with this last word
                    ngrams[front][word] = 1
                ngrams[front]["__counter__"] += 1
            else:  # first n-1 words of n-gram do not exist in the dictionary yet
                ngrams[front] = dict()
                ngrams[front][word] = 1
                ngrams[front]["__counter__"] = 1

        REMOVE_PUNCTUATION = args.punctuation  # User info
        print("Punctuation will " + ("" if REMOVE_PUNCTUATION else "not") + " be filtered out.")

        REMOVE_CAPITALIZATION = args.capitalization  # User info
        print("Capitalization will " + ("" if REMOVE_CAPITALIZATION else "not") + " be filtered out.")

        for fileNr, file in enumerate(files):  # Process each file one-by-one
            # 'latin1' is used to be able to read weird unicode characters without errors
            with open(file, "r", encoding='latin1') as f:
                fileLines = f.readlines()
            text = ' '.join(fileLines)  # Combine lines into single string

            printable = set(string.printable)  # Filter out weird unicode characters
            text = "".join(filter(lambda x: x in printable, text))

            # User info, prints e.g. "(3/10) Processing file 'this_is_a_text.txt'..."
            print("(" + str(fileNr + 1) + "/" + str(len(files)) + ") Processing file \'" + file + "\'...")

            # Filter out capitalization if flag is set
            if REMOVE_CAPITALIZATION:
                text = text.lower()

            # Filter punctuation and other formatting from text
            text = regex.sub("\n", "", text)
            if REMOVE_PUNCTUATION:
                text = regex.sub(r"[" + string.punctuation + "]+ +", " ", text)
                text = regex.sub(r" +[" + string.punctuation + "]+", " ", text)
            text = regex.sub(r"--", " ", text)
            text = regex.sub(r" +", " ", text)
            words = text.split(" ")

            # Generate the fronts (first n-1 words) for the n-grams, these are used for context
            for i in range(len(words)-(N-1)):
                fronts = [" ".join(words[i:i+offset]) for offset in range(N)]

                for j in range(N):
                    update_dictionary(fronts[j], words[i + j])

            # Track the maximum n-gram size for easier usage in the keyboard
            if "__max-depth__" in ngrams:
                ngrams["__max-depth__"] = max(ngrams["__max-depth__"], N)
            else:
                ngrams["__max-depth__"] = N

        # Write resulting dictionary to the output file
        json.dump(ngrams, open(OUTPUT_NAME, "w"))
        print("Created n-gram list \'" + OUTPUT_NAME + "\'.")

        # If flag -dictionary is set, also generate a unigram frequency list for the word correction
        if args.dictionary:
            path = os.path.join(basedirectory, "frequency_lists", LANGUAGE + ".txt")
            print("Creating frequency dictionary \'" + path + "\'")
            freq_file = ""
            if "" in ngrams:
                for word in ngrams[""]:
                    if not word == "__counter__":
                        # Add a line "<word> <frequency>" to the string:
                        freq_file += "{} {}\n".format(word, ngrams[""][word])
            # Write the frequency list to the dictionary file
            dict_output = open(path, "w")
            dict_output.write(freq_file)
            dict_output.close()

        print("Done!")

    @staticmethod
    def ensure_directory(path):
        if not os.path.isdir(path):
            print("Directory \'." + os.path.join(path) + "\' does not exist yet. Creating it now.")
            os.mkdir(path)


if __name__ == "__main__":
    CreateNgramFile.generate()
