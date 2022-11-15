

exact_match: num of matches in the sequence 
(if to be exact match from sequence to sequence, almost always be 0. Since currently not even doing well in prefix)
make sense to do so because no one push the model to learn hard for start token, which actually shows once, but can
make prefix_match = 0 if got wrong

use pack_padded_sequence and pad_packed_sequences gives me some improvement
but it needs to be used on cpu
so I check the device type in my code when running and only use the function when it's on cpu
it should contribute to some improvement in the performance, but running the model on cpu is too costly so I 
don't have the chance to fully run a model on cpu with these two functions

add a weight for the crossentropy: emphasizing on the <start token>, ideally to at least get a non-zero prefix-score

attention can be improved (turn loop into matrix operation)

