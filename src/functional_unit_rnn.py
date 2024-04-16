from torch import nn


class FunctionalUnitRNNClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            component_hidden_dim: int,
            component_layers: int,
            coordination_hidden_dim: int,
            coordination_layers: int,
            dropout: float,
            fc_hidden_dim: int,
            fc_layers: int,
            num_classes: int
    ):
        """
        Initializes an RNN with component and coordination functional units.
        - component RNN layers that take individual joints.
        - coordination RNN layers that take component output.
        - rnn_layers RNN layers stacked on each other.
        - Dropout layer.
        - fc_layers Linear layers stacked on each other.
        - Linear output layer for the classifier.

        Args:
            input_dim: The number of markers in each frame of the input sequence.
            component_hidden_dim: The hidden dimension of the RNN cell in the component units.
            component_layers: The number of RNN layers to stack for each component unit.
            coordination_hidden_dim: The hidden dimension of the RNN cell in the coordination units.
            coordination_layers: The number of RNN layers to stack for each coordination unit.
            fc_hidden_dim: The hidden dimension of the fully connected layer.
            fc_layers: The number of fully connected layers to stack.
            num_classes: The number of classes in the output.
        """
        super(FunctionalUnitRNNClassifier, self).__init__()
        pass

    def forward(self, x):
        """
        Performs a forward pass of the classifier.

        Args:
            x: A tensor of shape (B, L, self.input_dim, 3).
        """
        pass
