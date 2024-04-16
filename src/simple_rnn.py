from torch import nn


class SimpleRNNClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            rnn_hidden_dim: int,
            rnn_layers: int,
            dropout: float,
            fc_hidden_dim: int,
            fc_layers: int,
            num_classes: int
    ):
        """
        Initializes a simple RNN model:
        - rnn_layers RNN layers stacked on each other.
        - Dropout layer.
        - fc_layers Linear layers stacked on each other.
        - Linear output layer for the classifier.

        Args:
            input_dim: The number of markers in each frame of the input sequence.
            rnn_hidden_dim: The hidden dimension of the RNN cell.
            rnn_layers: The number of RNN layers to stack.
            fc_hidden_dim: The hidden dimension of the fully connected layer.
            fc_layers: The number of fully connected layers to stack.
            num_classes: The number of classes in the output.
        """
        super(SimpleRNNClassifier, self).__init__()
        # Initialize useful internal variables.
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.num_classes = num_classes
        # Initialize model layers.
        self.rnn_layers = nn.RNN(input_dim, rnn_hidden_dim, rnn_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = []
        for i in range(fc_layers):
            self.fc_layers.append(nn.Linear(rnn_hidden_dim, fc_hidden_dim))
        self.out_layer = nn.Linear(fc_hidden_dim, self.num_classes)

    def forward(self, x):
        """
        Performs a forward pass of the classifier.

        Args:
            x: A tensor of shape (B, L, self.input_dim, 3).
        """
        out, _ = self.rnn(x)  # Hidden state initialization defaults to 0s.
        out = self.dropout(out)
        for layer in self.fc_layers:
            out = layer(out)
        out = self.out_layer(out)
        return out
