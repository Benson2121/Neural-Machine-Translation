# Neural-Machine-Translation

## Introduction

### 1.Canadian Hansards
The main corpus for this assignment comes from the official records (Hansards) of the 36th Canadian Parliament, including debates from both the House of Representatives and the Senate. This corpus has been split into Training/ and Testing/ directories.
This data set consists of pairs of corresponding files (*.e is the English equivalent of the French *.f) in which every line is a sentence. Here, sentence alignment has already been performed for you. That is, the nth sentence in one file corresponds to the nth sentence in its corresponding file (e.g., line n in fubar.e is aligned with line n in fubar.f). Note that this data only consists of sentence pairs; many-to-one, many-to-many, and one-to-many alignments are not included.


### 2.Seq2seq
We will be implementing a simple seq2seq model, without attention, with single-headed attention, and with multi-headed attention based largely on the course material. You will train the models with teacher- forcing and decode using beam search. We will write it in [PyTorch version 1.13](https://pytorch.org/docs/1.13/), and Python version 3.10, which are the versions installed on the teach.cs servers. For those unfamiliar with PyTorch, we suggest you first read the [PyTorch tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).


### 3.Tensors and batches
PyTorch, like many deep learning frameworks, operate with tensors, which are multi-dimensional arrays.
When you work in PyTorch, you will rarely if ever work with just one bitext pair at a time. You’ll
instead be working with multiple sequences in one tensor, organized along one dimension of the batch.
This means that a pair of source and target tensors F and E actually correspond to multiple sequences.

We work with batches instead of individual sequences because a) backpropagating the average gradient over a batch tends to converge faster than single samples, and b) sample computations can be performed in parallel.
Learning to work with tensors can be difficult at first, but is integral to efficient computation. We suggest you read more about it in the [NumPy docs](https://numpy.org/doc/stable/user/basics.broadcasting.html), which PyTorch borrows for tensors.

### 4. Requirements
There are three changes to the seq2seq architectures that we make for this assignment. First, instead of scaled dot-product attention score, we’ll use the cosine similarity between vectors u and v.

The second relates to how we calculate the first hidden state for the decoder when we don’t use attention.
Recall that a bidirectional recurrent architecture processes its input in both directions separately: the
forward direction processes (x_1,x_2,...,x_S) whereas the backward direction processes (x_S,x_S−1,...,x_1).
The bidirectional hidden state concatenates the forward and backward hidden states for the same time, To ensure the decoder gets access to all input from both directions, you should initialize the first decoder state.

## Setup
1. Download the data from and extract it into the data/ directory.
2. Initialize a virtual environment and install the requirements.txt file.

## Part 1. Calculating BLEU scores

Modify bleu_score.py to be able to calculate BLEU scores on single reference and candidate strings.

Your functions will operate on sequences (e.g., lists) of tokens. These tokens could be the words themselves (strings) or an integer ID corresponding to the words. Your code should be agnostic to the type of token used, though you can assume that both the reference and candidate sequences will use tokens of the same type.

## Part 2. Building the encoder/decoder

You are expected to fill out a number of methods in encoder_decoder.py. These methods belong to sub-classes of the abstract base classes in abcs.py. The latter defines the abstract classes EncoderBase, DecoderBase, and EncoderDecoderBase, which implement much of the boilerplate code necessary to get a seq2seq model up and running. 

Though you are welcome to read and understand this code, it is not necessary to do so for this project. You will, however, need to read the doc strings in abcs.py to understand what you’re supposed to fill out in encoder_decoder.py.

1. **Encoder**

 - encoder_decoder.Encoder will be the concrete implementation of all encoders you will use. The encoder is always a multi-layer neural network with a bidirectional recurrent architecture. The encoder gets a batch of source sequences as input and outputs the corresponding sequence of hidden states from the last recurrent layer.


 - Encoder.forward_pass defines the structure of the encoder. For every model in PyTorch, the forward function defines how the model will run, and the forward function of every encoder or decoder will first clean up your input data and call forward pass to actually define the model structure. Now you need to implement the forward pass function that defines how your encoder will run.


 - Encoder.init_submodules(...) should be filled out to initialize a word embedding layer and a recurrent network architecture.

2. **Decoder without attention**

 - Encoder_decoder.DecoderWithoutAttention will be the concrete implementation of the decoders that do not use attention (so-called “transducer” models). Method implementations should thus be tailored to not use attention.


 - In order to feed the previous output into the decoder as input, the decoder can only process one step of input at a time and produce one output. Thus DecoderWithoutAttention is designed to process one slice of input at a time (though it will still be a batch of input for that given slice of time).


 - DecoderWithoutAttention.forward_pass defines the structure of the decoder. You will need to implement the forward pass function that defines how your decoder will run.


 - DecoderWithoutAttention.init_submodules(...) should be filled out to initialize a word embedding layer and a recurrent network architecture.


 - DecoderWithoutAttention.first_hidden_state(...) should be filled out to initialize the hidden state of the decoder. This is the hidden state that will be passed into the decoder at the first time step.


 - DecoderWithoutAttention.get_current_rnn_input(...) should be filled out to get the input to the decoder at the current time step. This is the input that will be passed into the decoder at the current time step.


 - DecoderWithoutAttention.get_current_hidden_state should be filled out to get the hidden state of the decoder at the current time step. This is the hidden state that will be passed into the decoder at the next time step.


 - DecoderWithoutAttention.get_current_logits(...) should be filled out to get the logits of the decoder at the current time step. This is the logits that will be passed into the decoder at the current time step.

3. **Decoder with (single-headed) attention**

 - Encoder_decoder.DecoderWithAttention will be the concrete implementation of the decoders that use attention. Method implementations should thus be tailored to use attention.


 - DecoderWithAttention.get_attention_scores should be filled out to get the attention scores of the decoder at the current time step. This is the attention scores that will be passed into the decoder at the current time step.


 - DecoderWithAttention.attend(...) should be filled out to get the attention scores of the decoder at the current time step. This is the attention scores that will be passed into the decoder at the current time step.

4. **Decoder with multi-head attention**

 - Encoder_decoder.DecoderWithMultiHeadAttention will be the concrete implementation of the decoders that use multi-head attention. Method implementations should thus be tailored to use multi-head attention.


 - DecoderWithMultiHeadAttention.init submodules(...) should be filled out to initialize a word embedding layer and a recurrent network architecture.


 - DecoderWithMultiHeadAttention.attend(...) should be filled out to get the attention scores of the decoder at the current time step. This is the attention scores that will be passed into the decoder at the current time step. You should initialize it to the hidden state of the encoder at the last time step.

5. **Encoder-Decoder**

 - Encoder_decoder.EncoderDecoder will be the concrete implementation of the encoder-decoder architecture. Method implementations should thus be tailored to use the encoder and decoder you implemented above.


 - EncoderDecoder.init submodules(...) initializes the encoder and decoder.


 - EncoderDecoder.get logits for teacher forcing(...) should be filled out to get the logits of the decoder at the current time step. This is the logits that will be passed into the decoder at the current time step.


 - EncoderDecoder.update beam(...) asks you to handle one iteration of a simplified version of the beam search. You should return the updated beam.

6. **Padding**

 - An important detail when dealing with sequences of input and output is how to deal with sequence lengths. Individual sequences within a batch may have different lengths, but the model expects all sequences to have the same length. Thus, we need to pad the sequences to the same length. This is done in the data loader, which you do not need to modify.

## Part 3. Training and evaluating the model

We will use ‘Weights and Biases’ [W&B] and ‘Tensorboard’ [TB] to visualize and log training. You will need to create an account on W&B and install the W&B python package. You will also need to install TB.

1. **Training the model**
 - Once you have completed the coding portion of the assignment, it is time you train your models. In order to do so in a reasonable amount of time, you’ll have to train your models using a machine with a GPU.

 - You are going to interface with your models using the script provided. Run the following code block line-by-line from your working directory. In order, it:

    - Builds maps between words and unique numerical identifiers for each language.
    - Splits the training data into a portion to train on and a hold-out portion.
    - Trains the encoder/decoder without attention and stores the model parameters.
    - Trains the encoder/decoder with single-headed attention and stores the model parameters.
    - Trains the encoder/decoder with multi-headed-attention and stores the model parameters.
    - Returns the average BLEU score of the encoder/decoder without attention on the test set.
    - Returns the average BLEU score of the encoder/decoder with single-headed attention on the test set.
    - Returns the average BLEU score of the encoder/decoder with multi-headed attention on the test set.

2. **Translation**

 - Ok, now we have the neural translation models. Let’s actually use them to translate some sentences.
 - In EncoderDecoder.translate(...), you are given a “raw” input sentence. You need to tokenize the sentence, convert the tokens into ordinal IDs, feed the IDs into your encoder-decoder model, and, finally, convert the output of the model into an actual sentence.
 - Translate the following three sentences using your models (without attention, with attention and with multi-head attention):
    - Toronto est une ville du Canada.
    - Les professeurs devraient bien traiter les assistants d’enseignement.
    - Les etudiants de l’Universite de Toronto sont excellents.

3. **Analysis**

 - In section 2 Translation Analysis of analysis.pdf, list all the translations. Then, describe the quality of those sentences. Can you observe any correlation with the model’s BLEU score? Include a brief discussion on your findings in analysis.pdf.

## Part 4. Bonus
You can get up to some bonus for this project. Some ideas:

 - Perform substantial qualitative data analysis of the translation results. Can you recognise some interesting patterns in your machine translation problem? Conduct statistical analysis to verify your findings, and use those findings to discuss the connection between statistical modelling and language.

 
 - Perform substantial data analysis of the error trends observed in each method you implement. This must go well beyond the basic discussion already included in the project.


 - Explore the effects of using different attention mechanisms for and include **attention visualization** of the different attention functions.

## Acknowledgements

The started code of this project is provided by:

University of Toronto 

CSC401: Natural Language Computing -- Winter 2023

Instructors: Annie En-Shiun Lee, Raeid Saqur, and Zining Zhu.

[Course Website](https://www.cs.toronto.edu/~raeidsaqur/csc401/)