from typing import List, Dict, Tuple, ClassVar
from torch import nn
import torch
import json


class FunctionalUnitRNNClassifier(nn.Module):
    """
    Represents an RNN classifier with functional units, implemented according to the paper:
        Zhang, Jianjing, et al. "Recurrent neural network for motion
        trajectory prediction in human-robot collaborative assembly."
        CIRP annals 69.1 (2020): 9-12.
    The RNN section has 4 coordination and 5 component functional units.

    Coordination units:
        - arm-arm
        - arm-spine
        - leg-leg
        - leg-spine
    Component units:
        - left arm
        - right arm
        - left leg
        - right leg
        - spine

    After the RNN with functional units, we have a standard dense network and classifier.

    Attributes:
        - joint_groups: (class attribute) The list of joint groups this RNN is operating on.
        - input_joint_map: A map of crops for each joint.
        - arm_arm_layers: Arm-arm coordination unit RNN.
        - arm_spine_layers: Arm-spine coordination unit RNN.
        - leg_leg_layers: Leg-leg coordination unit RNN.
        - leg_spine_layers: Leg-spine coordination unit RNN.
        - left_arm_layers: Left arm component unit RNN.
        - right_arm_layers: Right arm component unit RNN.
        - left_leg_layers: Left leg component unit RNN.
        - right_leg_layers: Right leg component unit RNN.
        - spine_layers: Spine component unit RNN.
        - dropout: Dropout layer between
        - fc_layers: Linear layers stacked on each other.
        - out_layer: Linear output layer for the classifier.
    """
    joint_groups: ClassVar[List[str]] = ['left arm', 'left leg', 'right arm', 'right leg', 'spine']
    input_joint_map: Dict[str, Tuple[int, int]]
    arm_arm_layers: nn.RNN
    arm_spine_layers: nn.RNN
    leg_leg_layers: nn.RNN
    leg_spine_layers: nn.RNN
    left_arm_layers: nn.RNN
    right_arm_layers: nn.RNN
    left_leg_layers: nn.RNN
    right_leg_layers: nn.RNN
    spine_layers: nn.RNN
    dropout: nn.Dropout
    fc_layers: nn.ModuleList
    out_layer: nn.Linear

    def __init__(
            self,
            joint_groups_file: str,
            coord_scale: float,
            coord_layers: int,
            comp_scale: float,
            comp_layers: int,
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
            joint_groups_file: Path to the file containing joint map information.
            coord_scale: Scale to apply to get hidden dimension = scale * input_dimension for each coordination unit.
            coord_layers: Number of RNN layers to stack for each coordination unit.
            comp_scale: Scale to apply to get hidden dimension = scale * input_dimension for each component unit.
            comp_layers: Number of RNN layers to stack for each component unit.
            fc_hidden_dim: Hidden dimension of the fully connected layer.
            fc_layers: Number of fully connected layers to stack.
            num_classes: Number of classes in the output.
        """
        super(FunctionalUnitRNNClassifier, self).__init__()
        # Process the joint groups into a mapping of the input.
        self.input_joint_map, joint_size_map = self._get_input_joint_map(joint_groups_file)
        # Set up the sizes for the coordination units.
        arm_arm_in = joint_size_map['left arm'] + joint_size_map['right arm']
        arm_arm_hidden = round(coord_scale * arm_arm_in)
        arm_spine_in = joint_size_map['left arm'] + joint_size_map['right arm'] + joint_size_map['spine']
        arm_spine_hidden = round(coord_scale * arm_spine_in)
        leg_leg_in = joint_size_map['left leg'] + joint_size_map['right leg']
        leg_leg_hidden = round(coord_scale * leg_leg_in)
        leg_spine_in = joint_size_map['left leg'] + joint_size_map['right leg'] + joint_size_map['spine']
        leg_spine_hidden = round(coord_scale * leg_spine_in)
        # Set up the sizes for the component units.
        left_arm_in = arm_arm_hidden + arm_spine_hidden + joint_size_map['left arm']
        left_arm_hidden = round(comp_scale * left_arm_in)
        right_arm_in = arm_arm_hidden + arm_spine_hidden + joint_size_map['right arm']
        right_arm_hidden = round(comp_scale * right_arm_in)
        left_leg_in = leg_leg_hidden + leg_spine_hidden + joint_size_map['left leg']
        left_leg_hidden = round(comp_scale * left_leg_in)
        right_leg_in = leg_leg_hidden + leg_spine_hidden + joint_size_map['right leg']
        right_leg_hidden = round(comp_scale * right_leg_in)
        spine_in = arm_spine_hidden + leg_spine_hidden + joint_size_map['spine']
        spine_hidden = round(comp_scale * spine_in)
        # Set up the layers for the coordination units.
        self.arm_arm_layers = nn.RNN(arm_arm_in, arm_arm_hidden, coord_layers, batch_first=True)
        self.arm_spine_layers = nn.RNN(arm_spine_in, arm_spine_hidden, coord_layers, batch_first=True)
        self.leg_leg_layers = nn.RNN(leg_leg_in, leg_leg_hidden, coord_layers, batch_first=True)
        self.leg_spine_layers = nn.RNN(leg_spine_in, leg_spine_hidden, coord_layers, batch_first=True)
        # Set up the layers for the component units.
        self.left_arm_layers = nn.RNN(left_arm_in, left_arm_hidden, comp_layers, batch_first=True)
        self.right_arm_layers = nn.RNN(right_arm_in, right_arm_hidden, comp_layers, batch_first=True)
        self.left_leg_layers = nn.RNN(left_leg_in, left_leg_hidden, comp_layers, batch_first=True)
        self.right_leg_layers = nn.RNN(right_leg_in, right_leg_hidden, comp_layers, batch_first=True)
        self.spine_layers = nn.RNN(spine_in, spine_hidden, comp_layers, batch_first=True)
        # Set up the dropout layer.
        self.dropout = nn.Dropout(dropout)
        # Set up the dense layers.
        fc_in = left_arm_hidden + right_arm_hidden + left_leg_hidden + right_leg_hidden + spine_hidden
        self.fc_layers = nn.ModuleList([nn.Linear(fc_in, fc_hidden_dim)
                                        for _ in range(fc_layers)])
        # Set up the output layer.
        self.out_layer = nn.Linear(fc_hidden_dim, num_classes)

    def _get_input_joint_map(self, joint_groups_file: str) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int]]:
        """
        Gets a map of [start: end] crops of the input dimension for each joint in the joint map.

        Args:
            joint_groups_file: The path to the file containing the joint map information.

        Returns:
            A map of crops for each joint.
            A map of sizes for each joint.
        """
        input_joint_map = {}
        joint_size_map = {}
        # Read the contents of the JSON file.
        with open(joint_groups_file, 'r') as f:
            joint_map = json.load(f)
        # Sort the joint groups we will be working with.
        self.joint_groups.sort()
        # For each joint group, create the crop map.
        i = 0
        for joint_group in self.joint_groups:
            n_markers = len(joint_map[joint_group])
            step = n_markers * 3
            input_joint_map[joint_group] = (i, i + step)
            joint_size_map[joint_group] = step
            i += step
        return input_joint_map, joint_size_map

    def forward(self, x):
        """
        Performs a forward pass of the classifier.

        Args:
            x: A tensor of shape (B, L, self.input_dim).
        """
        # Split the input up into the joint groups.
        start, end = self.input_joint_map['left arm']
        left_arm = x[:, :, start: end]
        start, end = self.input_joint_map['right arm']
        right_arm = x[:, :, start: end]
        start, end = self.input_joint_map['left leg']
        left_leg = x[:, :, start: end]
        start, end = self.input_joint_map['right leg']
        right_leg = x[:, :, start: end]
        start, end = self.input_joint_map['spine']
        spine = x[:, :, start: end]
        # Apply the coordination units.
        arm_arm_cat = torch.cat((left_arm, right_arm), dim=-1)
        arm_arm_out, _ = self.arm_arm_layers(arm_arm_cat)
        arm_spine_cat = torch.cat((left_arm, right_arm, spine), dim=-1)
        arm_spine_out, _ = self.arm_spine_layers(arm_spine_cat)
        leg_leg_cat = torch.cat((left_leg, right_leg), dim=-1)
        leg_leg_out, _ = self.leg_leg_layers(leg_leg_cat)
        leg_spine_cat = torch.cat((left_leg, right_leg, spine), dim=-1)
        leg_spine_out, _ = self.leg_spine_layers(leg_spine_cat)
        # Apply the component units.
        left_arm_cat = torch.cat((left_arm, arm_arm_out, arm_spine_out), dim=-1)
        left_arm_out, _ = self.left_arm_layers(left_arm_cat)
        right_arm_cat = torch.cat((right_arm, arm_arm_out, arm_spine_out), dim=-1)
        right_arm_out, _ = self.right_arm_layers(right_arm_cat)
        left_leg_cat = torch.cat((left_leg, leg_leg_out, leg_spine_out), dim=-1)
        left_leg_out, _ = self.left_leg_layers(left_leg_cat)
        right_leg_cat = torch.cat((right_leg, leg_leg_out, leg_spine_out), dim=-1)
        right_leg_out, _ = self.right_leg_layers(right_leg_cat)
        spine_cat = torch.cat((spine, arm_spine_out, leg_spine_out), dim=-1)
        spine_out, _ = self.spine_layers(spine_cat)
        # Concatenate the outputs of the functional units.
        rnn_out = torch.cat((left_arm_out, left_leg_out, right_arm_out, right_leg_out, spine_out), dim=-1)
        # Average the outputs for each element in the sequence.
        rnn_out = torch.mean(rnn_out, dim=1)
        # Apply dropout after RNN layer.
        out = self.dropout(rnn_out)
        # Run through dense layers.
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
        # Run through the final output layer to get logits.
        out = self.out_layer(out)
        return out
