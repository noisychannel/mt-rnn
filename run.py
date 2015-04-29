#!/usr/bin/env python

# TODO:
# 1. Get the phrase table and extract all phrases (Use the version from Phrase translation)
# 2. Get vocabulary : Keep the top 10k words and map the rest to UNK
# 3. Apply the UNK changes to the phrase table
# 4. Get the embeddings for the modified corpus
# 5. Shuffle the phrase pairs from the phrase table and split into train and dev sets (80/20)
# 6. Form mini batches from the training data
# 7. What is the structure of a phrase vector ? What is a good way of selecting these ? Draw inspiration from the sentence selection in the Theano example
# 8. Only apply averaged updates after a mini-batch is complete
# 9. Run for several epochs. The most successful epoch is the one which maximin
# 10. Retain best model in the form of a pickle dump so that it can be loaded by the feature function in moses

import sys
import os
import argparse


