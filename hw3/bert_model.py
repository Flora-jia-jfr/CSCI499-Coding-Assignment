# import torch
# from transformers import AutoTokenizer, EncoderDecoderModel, BertGenerationEncoder, BertGenerationTokenizer, \
#     BertGenerationDecoder, BertTokenizer
#
#
# # leverage checkpoints for Bert2Bert model...
# # use BERT's cls token as BOS token and sep token as EOS token
# encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased", bos_token_id=101, eos_token_id=102)
# # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
# decoder = BertGenerationDecoder.from_pretrained("bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
# bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
#
# # create tokenizer...
# tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
#
# input_ids = tokenizer('This is a long article to summarize', add_special_tokens=False, return_tensors="pt").input_ids
# # labels = tokenizer('This is a short summary', return_tensors="pt").input_ids
# # print("labels: ", labels)
# labels = torch.tensor([[3, 16], [4,19], [5,17]])
#
#
# # train...
# loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
# loss.backward()
#
#


from transformers import BertGenerationTokenizer, BertGenerationEncoder, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertGenerationEncoder.from_pretrained("bert-large-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape, last_hidden_states)