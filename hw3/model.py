# IMPLEMENT YOUR MODEL CLASS HERE
import torch.nn
import torch.nn as nn


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
        # print("batch_size for encoder: ", batch_size)
        embedded = self.embedding_layer(encoder_input.permute(1, 0))  # change order
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        # print("encoder embedded: ", embedded.shape)
        encoder_outputs, (encode_hidden_state, encoder_cell_state) = self.lstm_encoder(embedded, (h_0, c_0))
        return encoder_outputs, (encode_hidden_state, encoder_cell_state)


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
        # print("decode_input: ", decoder_input.shape)
        decoder_action_input = torch.unsqueeze(decoder_input[:, 0], dim=0)
        decoder_target_input = torch.unsqueeze(decoder_input[:, 1], dim=0)
        # print("decoder_action_input: ", decoder_action_input.shape)
        # print("decoder_target_input: ", decoder_target_input.shape)
        action_embedded = self.action_embedding(decoder_action_input) # seq_len, batch, 2
        target_embedded = self.target_embedding(decoder_target_input)
        # print("target embedded: ", target_embedded.shape)
        embedded = torch.cat((action_embedded, target_embedded), dim=-1)
        # print("embedded.shape: ", embedded.shape)
        # print("hidden.shape: ", hidden.shape)
        # print("cell.shape: ", cell.shape)
        decoder_outputs, (decode_hidden_state, decoder_cell_state) = self.lstm_decoder(embedded, (hidden, cell))
        # print("decoder_outputs: ", decoder_outputs.shape)
        return decoder_outputs, (decode_hidden_state, decoder_cell_state)


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, encoder, decoder, num_actions, num_targets, hidden_dim):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.hidden2action = torch.nn.Linear(hidden_dim, num_actions)
        self.hidden2target = torch.nn.Linear(hidden_dim, num_targets)

    def forward(self, encoder_input, decoder_target, teacher_forcing=False):
        batch_size = encoder_input.shape[0]  # 512
        assert batch_size == decoder_target.shape[0]  # assert input batch == output batch
        # print("batch_size for encoderDecoder: ", batch_size)
        # print("decoder_target shape: ", decoder_target.shape)  # should be [batch_size, max_len, 2] => [512, 115, 2]
        max_len = decoder_target.shape[1]
        action_outputs = torch.zeros(batch_size, max_len, self.num_actions)
        target_outputs = torch.zeros(batch_size, max_len, self.num_targets)
        encoder_output, (encoder_hidden_state, encoder_cell_state) = self.encoder(encoder_input)
        # print("(encoder_hidden_state, encoder_cell_state): ", encoder_hidden_state.shape, encoder_cell_state.shape)
        # TODO: should change this to null initialization
        decoder_input = torch.zeros(batch_size, 2, dtype=int)
        # decoder_input = decoder_target[:, 0, :]  # [512, 2]
        # print("decoder_target: ", decoder_target.shape)
        # print("decoder input: ", decoder_input.shape)
        hidden, cell = encoder_hidden_state, encoder_cell_state
        for i in range(max_len):
            # print("decoder input: ", decoder_input.shape)
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            predicted_action = self.hidden2action(output)
            predicted_target = self.hidden2target(output)
            # print("predicted_action:", predicted_action.shape)
            # print("predicted_target:", predicted_target.shape)
            action_outputs[:, i] = predicted_action
            target_outputs[:, i] = predicted_target
            # print(action_outputs.shape)
            # print(target_outputs.shape)
            # print(torch.squeeze(torch.argmax(predicted_action, dim=2)).shape)
            # print(torch.squeeze(torch.argmax(predicted_target, dim=2)).shape)
            if teacher_forcing:
                decoder_input = decoder_target[:, i, :]
                # print("teaching forcing decoder input: ", decoder_input.shape)
            else:
                predicted_pair = torch.cat((torch.squeeze(torch.argmax(predicted_action, dim=2)),
                                           torch.squeeze(torch.argmax(predicted_target, dim=2)))).reshape(batch_size, 2)
                # TODO: check
                decoder_input = predicted_pair
                # print("for next decoder_input(student):", predicted_pair.shape, predicted_pair)
        # print("action_outputs: ", action_outputs.shape)
        # print("target_outputs: ", target_outputs.shape)
        return action_outputs, target_outputs



