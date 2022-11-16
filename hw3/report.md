# Model Report

Furong(Flora) Jia | CSCI 499 Coding Assignment 3

## Implementation Detail

### Preprocessing Data
* pack instructions of an episode together to one sequence, add `<start>` token in the front and
  `<end>` token around the long instruction (length of sequence is around 140 or something)
* add additional tokens for output labels: `<start> = 0`, `<pad> = 1`, `<stop>=2`, make the seq_len for
  labels the max_len of labels in one episode, add `<start>` and `<end>` token and pad it with `<pad>`

### Evaluation
implement `prefix_match` and `exact_match`: exact_match is a more generous version of prefix_match
* prefix_match: need all the predictions before it also to be true to be counted
* exact_match: actual number of matches (no need for the previous predictions to be true)

I implemented the exact_match because I got a bunch of zeros for the prefix_match at the start, and
I want to be less strict to my model to see how it predicts in the whole sequence. 
Also, I add a weight tensor for the CrossEntropy trying to make it take more efforts to predict `<start>`,
so it won't get a prefix_match of 0 for getting the `<start>` wrong (can't tell anything in that case)
I take the exact_match as a similar version of the accuracy in hw1 (since it also compares individual
predictions), despite it's actually a sequential output here.

### Model implementation
I choose LSTM for encoder and decoders.

**Encoder**

It first embeds instructions to word embeddings, and then use the LSTM to encoder the long instruction
into encoder_output(which contain all hidden states of the sequence), hidden_state, and cell_state.
I used `pack_padded_sequence` and `pad_packed_sequence` to improvement the performance

**Decoder**

It embeds the action and target labels from the previous results, and then concatenate the two embeddings.
The concatenated representation is then processed by the LSTM decoder to get decoded representations

**EncoderDecoder**

The basic version of encoderDecoder is like a wrapper for the encoder and decoder.
It first gets the output and hidden from the encoder, and then went through a loop containing decoders to
get the decoded representation. 

In the loop, the representation is then interpreted into actions and targets
(logits) by two separate linear model. In training mode, `teacher_forcing` is set to True and the next decoder
will take in the correct label for the next iteration; in validation mode, the model will take its predicted
results into the next iteration of decoder and operate based on that. 

In the end, sequences of predictions are returned from the model as logits over each class.
They are turned into actual predictions by taking `argmax` in `train.py`

**EncoderDecoder with Attention**
If `args.model==attention`, it will compute the softmax score for each encoder hidden state (namely,
for the hidden state of each timeframe) in the `getAttention` function in `EncoderDecoder` class.

It is implemented by first repeating the decoder hidden state to the same length as the encoder_output
(seq_len, batch_size, hidden_dim), and then concat the hidden_dim for the two tensors. (Make the shape
(seq_len, batch_size, 2*hidden_dim)). 

Then it use a fully-connected layer to get the score for each concatenated hidden state. 
(map from 2*hidden_dim to 1 -- a single number)

$x + y$

the 


## Model Performance

I add another argument `--model` to determine which kind of model to work on,

### EncoderDecoder (Vanilla)

### Attention
**Discuss your encoder-decoder attention choices (e.g., flat or hierarchical, recurrent cell used, etc.)**

**Discuss the attention mechanism you implemented for the encoder-decoder model using the taxonomy we discussed in class.**
### BERT (replacing encoder)
idea is to get a better representation of the whole sequence -- less forgetness for the instructions at the front







exact_match: num of matches in the sequence 
(if to be exact match from sequence to sequence, almost always be 0. Since currently not even doing well in prefix)
make sense to do so because no one push the model to learn hard for start token, which actually shows once, but can
make prefix_match = 0 if got wrong

when i realize i have bug in my evluation metrics and it actually could perform better, its too late

weird that my vanilla outperforms attention

use pack_padded_sequence and pad_packed_sequences gives me some improvement
but it needs to be used on cpu
so I check the device type in my code when running and only use the function when it's on cpu
it should contribute to some improvement in the performance, but running the model on cpu is too costly so I 
don't have the chance to fully run a model on cpu with these two functions

add a weight for the crossentropy: emphasizing on the <start token>, ideally to at least get a non-zero prefix-score

attention can be improved (turn loop into matrix operation)

