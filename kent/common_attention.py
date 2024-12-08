import torch.nn as nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention mechanism to compute attention weights.
    """

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        # Input to attn is concatenated (encoder_hidden_dim * 2) + decoder_hidden_dim
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, decoder_hidden_dim]
        # encoder_outputs: [batch_size, src_len, encoder_hidden_dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state src_len times for concatenation
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, decoder_hidden_dim]
        
        # Concatenate hidden and encoder_outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, decoder_hidden_dim]
        
        # Compute attention weights
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return F.softmax(attention, dim=1)

class EncoderWithAttention(nn.Module):
    """
    Encoder with attention mechanism.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(EncoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass of the encoder.

        Args:
            src: [batch_size, src_len] - Input sequences.

        Returns:
            outputs: [batch_size, src_len, hidden_dim] - Encoder outputs for attention.
            hidden: [n_layers, batch_size, hidden_dim] - Final hidden state.
        """
        # Embed and apply dropout
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]

        # Pass through GRU
        outputs, hidden = self.rnn(embedded)  # outputs: [batch_size, src_len, hidden_dim]
                                              # hidden: [n_layers, batch_size, hidden_dim]

        return outputs, hidden  # Return both for attention

class DecoderWithAttention(nn.Module):
    """
    Decoder with attention mechanism.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention, encoder_hidden_dim):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attention = attention

        # Add a layer to reduce encoder hidden states to match decoder hidden_dim
        self.reduce_hidden = nn.Linear(encoder_hidden_dim * 2, hidden_dim)  # Project bidirectional hidden states

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(
            emb_dim + (encoder_hidden_dim * 2),  # Input includes weighted context
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear((encoder_hidden_dim * 2) + hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        Forward pass for Decoder with Attention.
        Args:
        - input: [batch_size] - Current target token.
        - hidden: [n_layers, batch_size, hidden_dim] - Decoder hidden state.
        - encoder_outputs: [batch_size, src_len, encoder_hidden_dim * 2] - Encoder outputs.
        """

        print(f"DEBUG: Encoder outputs: {encoder_outputs.shape}, Decoder hidden state: {hidden.shape}")

        # Reduce hidden state dimensionality
        if hidden.size(2) != self.hidden_dim:  # Only apply if dimensions don't match
            hidden = torch.tanh(self.reduce_hidden(hidden.permute(1, 0, 2)))  # [batch_size, n_layers, hidden_dim]
            hidden = hidden.permute(1, 0, 2)  # [n_layers, batch_size, hidden_dim]

        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        # Attention mechanism
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, encoder_hidden_dim * 2]

        # Concatenate context and embedded input
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [batch_size, 1, emb_dim + encoder_hidden_dim * 2]

        output, hidden = self.rnn(rnn_input, hidden)  # output: [batch_size, 1, hidden_dim]
        output = output.squeeze(1)  # [batch_size, hidden_dim]
        weighted = weighted.squeeze(1)  # [batch_size, encoder_hidden_dim * 2]

        # Compute final output prediction
        prediction = self.fc_out(torch.cat((output, weighted, embedded.squeeze(1)), dim=1))  # [batch_size, output_dim]

        return prediction, hidden

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the Seq2Seq model with attention.

        Args:
            src: [batch_size, src_len] - Source sequences.
            trg: [batch_size, trg_len] - Target sequences.
            teacher_forcing_ratio: Probability to use teacher forcing.

        Returns:
            outputs: [trg_len, batch_size, output_dim] - Decoder outputs.
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Initialize output tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Pass source through encoder
        encoder_outputs, hidden = self.encoder(src)

        # Initialize the first input as the <bos> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Decode using attention
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # Store the output
            outputs[t] = output

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs
