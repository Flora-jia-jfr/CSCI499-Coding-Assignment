# IMPLEMENT YOUR MODEL CLASS HERE
import torch.nn
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # embedding
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers)

    def forward(self, encoder_input):
        # print("--------------IN ENCODER--------------")
        batch_size = encoder_input.shape[0]
        # print("batch_size for encoder: ", batch_size) # [512, 519]
        embedded = self.embedding_layer(encoder_input.permute(1, 0))  # change order
        # print("encoder embedded: ", embedded.shape)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        if self.device == torch.device("cpu"):
            seq_len = torch.tensor([torch.count_nonzero(encoder_input[i]) for i in range(batch_size)])
            packed_input = pack_padded_sequence(embedded, seq_len, enforce_sorted=False)
            packed_encoder_outputs, (encode_hidden_state, encoder_cell_state) = self.lstm_encoder(packed_input, (h_0, c_0))
            padded_encoder_outputs, _ = pad_packed_sequence(packed_encoder_outputs)
            return padded_encoder_outputs.to(self.device), (encode_hidden_state.to(self.device), encoder_cell_state.to(self.device))
        else:
            encoder_outputs, (encode_hidden_state, encoder_cell_state) = self.lstm_encoder(embedded, (h_0, c_0))
            return encoder_outputs, (encode_hidden_state, encoder_cell_state)


class BERTEncoder(nn.Module):
    """
    Encode a sequence of tokens using pre-trained bert
    """

    def __init__(self, device, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        assert hidden_dim == 768
        self.hidden_dim = hidden_dim
        self.bert = BertModel.from_pretrained('bert-base-cased')

    def forward(self, encoder_input):
        output, hidden = self.bert(input_ids=encoder_input, return_dict=False)
        cell_state = torch.zeros(hidden.shape)
        return output, (hidden, cell_state)


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, embedding_dim, hidden_dim, num_layers, num_actions, num_targets):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # embedding for actions and targets
        self.action_embedding = torch.nn.Embedding(num_actions, embedding_dim)
        self.target_embedding = torch.nn.Embedding(num_targets, embedding_dim)
        # LSTM decoder
        self.lstm_decoder = nn.LSTM(2*embedding_dim, hidden_dim, num_layers)  # seq_len, batch_size, hidden_dimension


    def forward(self, decoder_input, hidden, cell):  # decoder_input: [batch_size, 2]
        # print("--------------IN DECODER--------------")
        batch_size = decoder_input.shape[0]
        # print("batch_size for decoder: ", batch_size)
        # print("decode_input: ", decoder_input.shape, decoder_input)
        decoder_action_input = torch.unsqueeze(decoder_input[:, 0], dim=0)
        decoder_target_input = torch.unsqueeze(decoder_input[:, 1], dim=0)
        # print("decoder_action_input: ", decoder_action_input.shape, decoder_action_input)
        # print("decoder_target_input: ", decoder_target_input.shape, decoder_target_input)
        action_embedded = self.action_embedding(decoder_action_input.to(self.device)) # seq_len, batch, 2
        target_embedded = self.target_embedding(decoder_target_input.to(self.device))
        # print("target embedded: ", target_embedded.shape)
        embedded = torch.cat((action_embedded, target_embedded), dim=-1)
        # print("embedded.shape: ", embedded.shape)
        # print("hidden.shape: ", hidden.shape)
        # print("cell.shape: ", cell.shape)
        decoder_outputs, (decode_hidden_state, decoder_cell_state) = self.lstm_decoder(embedded, (hidden, cell))
        # print("decoder_outputs: ", decoder_outputs.shape)
        return decoder_outputs, (decode_hidden_state, decoder_cell_state)

#
# class EncoderDecoder(nn.Module):
#     """
#     Wrapper class over the Encoder and Decoder.
#     TODO: edit the forward pass arguments to suit your needs
#     """
#
#     def __init__(self, device, encoder, decoder, num_actions, num_targets, hidden_dim):
#         super().__init__()
#         self.device = device
#         self.encoder = encoder
#         self.decoder = decoder
#         self.num_actions = num_actions
#         self.num_targets = num_targets
#         self.hidden_dim = hidden_dim
#         self.hidden2action = torch.nn.Linear(hidden_dim, num_actions)
#         self.hidden2target = torch.nn.Linear(hidden_dim, num_targets)
#
#     def forward(self, encoder_input, decoder_target, teacher_forcing=False):
#         batch_size = encoder_input.shape[0]  # 512
#         assert batch_size == decoder_target.shape[0]  # assert input batch == output batch
#         # print("batch_size for encoderDecoder: ", batch_size)
#         # print("decoder_target: ", decoder_target.shape, decoder_target)  # should be [batch_size, max_len, 2] => [512, 115, 2]
#         max_len = decoder_target.shape[1]
#         action_outputs = torch.zeros(batch_size, max_len, self.num_actions)
#         target_outputs = torch.zeros(batch_size, max_len, self.num_targets)
#         encoder_output, (encoder_hidden_state, encoder_cell_state) = self.encoder(encoder_input)
#         # print("encoder_output.shape: ", encoder_output.shape)
#         # print("(encoder_hidden_state, encoder_cell_state): ", encoder_hidden_state.shape, encoder_cell_state.shape)
#         decoder_input = torch.zeros(batch_size, 2, dtype=int)
#         # decoder_input = decoder_target[:, 0, :]  # [512, 2]
#         # print("decoder_target: ", decoder_target.shape)
#         # print("decoder input: ", decoder_input.shape)
#         hidden, cell = encoder_hidden_state, encoder_cell_state
#         for i in range(max_len):
#             # print("decoder input: ", decoder_input.shape)
#             output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
#             # hidden and output is the same, since seq_len = 1 for decoder
#             predicted_action = self.hidden2action(hidden)
#             predicted_target = self.hidden2target(hidden)
#             # print("predicted_action:", predicted_action.shape)
#             # print("predicted_target:", predicted_target.shape)
#             action_outputs[:, i] = predicted_action  # TODO: output of bound
#             target_outputs[:, i] = predicted_target
#             # print(action_outputs.shape)
#             # print(target_outputs.shape)
#             # print(torch.squeeze(torch.argmax(predicted_action, dim=2)).shape)
#             # print(torch.squeeze(torch.argmax(predicted_target, dim=2)).shape)
#             if teacher_forcing:
#                 decoder_input = decoder_target[:, i, :]
#                 print("teaching forcing decoder input: ", decoder_input.shape)
#             else:
#                 predicted_pair = torch.cat((torch.squeeze(torch.argmax(predicted_action, dim=2)),
#                                            torch.squeeze(torch.argmax(predicted_target, dim=2))), dim=1)
#                 # TODO: check
#                 decoder_input = predicted_pair
#
#         # print("action_outputs: ", action_outputs.shape)
#         # print("target_outputs: ", target_outputs.shape)
#         return action_outputs, target_outputs


# Attention:
class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder with Attention.
    """

    def __init__(self, device, encoder, decoder, num_actions, num_targets, hidden_dim, attention=False):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.hidden2action = torch.nn.Linear(hidden_dim, num_actions)
        self.hidden2target = torch.nn.Linear(hidden_dim, num_targets)
        self.attention = attention
        # add fully connected layer used in attention
        self.fc = torch.nn.Linear(hidden_dim * 2, 1) # map from the concat of two hidden states to a single number
        # add softmax layer
        self.softmax = nn.Softmax(dim=0)


    def getAttention(self, decoder_hidden, encoder_out):
        batch_size = encoder_out.shape[1]
        assert batch_size == decoder_hidden.shape[1]
        decoder_hidden = decoder_hidden.squeeze(0)
        # torch.Size([519, 512, 256]) -- encoder_out
        seq_len = encoder_out.shape[0]
        # torch.Size([1, 512, 256]) -- decoder_hidden
        logits = torch.zeros(seq_len, batch_size, dtype=float)

        # [batch, seq_len]
        for i in range(seq_len):
            encoder_hidden_i = encoder_out[i]
            # print("encoder_hidden_i: ", encoder_hidden_i.shape)  # torch.Size([512, 256])
            # print("decoder_hidden: ", decoder_hidden.shape)  # torch.Size([512, 256])
            concat_hidden_state = torch.cat((encoder_hidden_i, decoder_hidden), dim=1)
            # print("concat_hidden_state: ", concat_hidden_state)
            score = self.fc(concat_hidden_state).squeeze(1)
            # print("score: ", score.shape)  # torch.Size([512])
            logits[i, :] = score
        # print("logits: ", logits.shape, logits)  # torch.Size([519, 512])
        softmax_logits = self.softmax(logits).float()
        # print("softmax_logits: ", softmax_logits.shape, softmax_logits)
        # print(softmax_logits[:, 0].sum())
        return softmax_logits


    def forward(self, encoder_input, decoder_target, teacher_forcing=False):
        batch_size = encoder_input.shape[0]  # 512
        assert batch_size == decoder_target.shape[0]  # assert input batch == output batch
        # print("batch_size for encoderDecoder: ", batch_size)
        # print("decoder_target: ", decoder_target.shape, decoder_target)
        # should be [batch_size, max_len, 2] => [512, 115, 2]
        max_len = decoder_target.shape[1]
        action_outputs = torch.zeros(batch_size, max_len, self.num_actions)
        target_outputs = torch.zeros(batch_size, max_len, self.num_targets)
        # torch.Size([519, 512, 256]) torch.Size([1, 512, 256]) torch.Size([1, 512, 256])
        encoder_output, (encoder_hidden_state, encoder_cell_state) = self.encoder(encoder_input)
        decoder_input = torch.zeros(batch_size, 2, dtype=int)
        # decoder_input = decoder_target[:, 0, :]  # [512, 2]
        # print("decoder_target: ", decoder_target.shape)
        # print("decoder input: ", decoder_input.shape)
        hidden, cell = encoder_hidden_state, encoder_cell_state
        for i in range(max_len):
            # print("decoder input: ", decoder_input.shape)
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            # torch.Size([519, 512])
            if self.attention:
                softmax_logits = self.getAttention(hidden, encoder_output).unsqueeze(2).to(self.device) # decoder_hiddem, encoder_output
                # print("softmax_logits: ", softmax_logits.shape) -- torch.Size([519, 512])
                # torch.Size([519, 512, 256]) -- encoder_output
                batch_first_softmax_logits = softmax_logits.permute(1, 0, 2)  # [512, 519, 1]
                batch_first_encoder_output = encoder_output.permute(1, 2, 0)  # [512, 256, 519]
                # print("batch_first_softmax_logits: ", batch_first_softmax_logits.shape)
                # print("batch_first_encoder_output: ", batch_first_encoder_output.shape)
                # TODO: calculate weighted hidden state to be inputted into linear map DEBUG
                weighted_hidden_state = torch.bmm(batch_first_encoder_output, batch_first_softmax_logits).float().squeeze(2)
                # print("weighted_hidden_state: ", weighted_hidden_state.shape)  # [512, 256]
            else:
                weighted_hidden_state = hidden
            predicted_action = self.hidden2action(weighted_hidden_state)
            predicted_target = self.hidden2target(weighted_hidden_state)
            # print("predicted_action:", predicted_action.shape)
            # print("predicted_target:", predicted_target.shape)
            action_outputs[:, i] = predicted_action  # TODO: output of bound
            target_outputs[:, i] = predicted_target
            # print(action_outputs.shape)
            # print(target_outputs.shape)
            # print(torch.squeeze(torch.argmax(predicted_action, dim=2)).shape)
            # print(torch.squeeze(torch.argmax(predicted_target, dim=2)).shape)
            if teacher_forcing:
                decoder_input = decoder_target[:, i, :]
                # print("teaching forcing decoder input: ", decoder_input.shape, decoder_target)
            else:
                # print("action logits: ", predicted_action)
                # print("target logits: ", predicted_target)
                # print("predicted action: ", torch.argmax(predicted_action, dim=1).shape, torch.argmax(predicted_action, dim=1))
                # print("predicted target: ", torch.argmax(predicted_target, dim=1).shape, torch.argmax(predicted_target, dim=1))
                predicted_pair = torch.stack((torch.argmax(predicted_action, dim=1),
                                           torch.argmax(predicted_target, dim=1)), dim=-1)
                # TODO: check
                decoder_input = predicted_pair
                # print("predicted pair: ", predicted_pair.shape, predicted_pair)
                # print("teaching forcing decoder input: ", decoder_target[:, i, :].shape, decoder_target[:, i, :])
                # print("for next decoder_input(student):", predicted_pair.shape, predicted_pair)
        # print("action_outputs: ", action_outputs.shape)
        # print("target_outputs: ", target_outputs.shape)
        return action_outputs, target_outputs
