from torch import nn
import torch


class SimpleRNNClassifier(nn.Module):
    """
    Represents a simple RNN classifier, consisting of:
        - rnn_layers RNN layers stacked on each other.
        - 1 dropout layer.
        - fc_layers Linear layers stacked on each other.
        - 1 linear output layer for the classifier.

    Attributes:
        rnn_layers: RNN layers stacked on each other.
        dropout: The dropout layer between the RNN and Linear layers.
        fc_layers: Linear layers stacked on each other.
        out_layer: Linear output layer for the classifier.
    """
    rnn_layers: nn.RNN
    dropout: nn.Dropout
    fc_layers: nn.ModuleList
    out_layer: nn.Linear

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
        Initializes a SimpleRnnClassifier object with the given specifications.

        Args:
            input_dim: The side of the flattened data dimension for each frame, the number of markers * 3.
            rnn_hidden_dim: The hidden dimension of the RNN cell.
            rnn_layers: The number of RNN layers to stack.
            fc_hidden_dim: The hidden dimension of the fully connected layer.
            fc_layers: The number of fully connected layers to stack.
            num_classes: The number of classes in the output.
        """
        super(SimpleRNNClassifier, self).__init__()
        self.rnn_layers = nn.RNN(input_dim, rnn_hidden_dim, rnn_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = nn.ModuleList([nn.Linear(rnn_hidden_dim, fc_hidden_dim)
                                        for _ in range(fc_layers)])
        self.out_layer = nn.Linear(fc_hidden_dim, num_classes)

    def forward(self, x):
        """
        Performs a forward pass of the classifier.

        Args:
            x: A tensor of shape (B, L, self.input_dim).
        """
        # Run through rnn layers, hidden state defaults to 0s.
        out, _ = self.rnn_layers(x)
        # Average the outputs for each element in the sequence.
        out = torch.mean(out, dim=1)
        # Apply dropout after RNN layer.
        out = self.dropout(out)
        # Run through dense layers.
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
        # Run through final output layer to get logits.
        out = self.out_layer(out)
        return out
