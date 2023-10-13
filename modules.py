from graph_nets import _base
from graph_nets import blocks
import tensorflow as tf


class InteractionNetwork(_base.AbstractModule):
  """Implementation of an Interaction Network.

  An interaction networks computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incomming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This model does not update the graph globals, and they are allowed to be
  `None`.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               global_model_fn,
               reducer=tf.math.unsorted_segment_sum,
               name="interaction_network"):
    """Initializes the InteractionNetwork module.

    Args:
      edge_model_fn: A callable that will be passed to `EdgeBlock` to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.EdgeBlock` for details), and the shape of the
        output of this module must match the one of the input nodes, but for the
        first and last axis.
      node_model_fn: A callable that will be passed to `NodeBlock` to perform
        per-node computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.NodeBlock` for details).
      reducer: Reducer to be used by NodeBlock to aggregate edges. Defaults to
        tf.math.unsorted_segment_sum.
      name: The module name.
    """
    super(InteractionNetwork, self).__init__(name=name)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn,
          use_receiver_nodes=False,
          use_sender_nodes=False,
          use_globals=False)

  def _build(self, graph):
    """Connects the InterationNetwork.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s. `graph.globals` can be
        `None`. The features of each node and edge of `graph` must be
        concatenable on the last axis (i.e., the shapes of `graph.nodes` and
        `graph.edges` must match but for their first and last axis).

    Returns:
      An output `graphs.GraphsTuple` with updated edges and nodes.

    Raises:
      ValueError: If any of `graph.nodes`, `graph.edges`, `graph.receivers` or
        `graph.senders` is `None`.
    """
    return self._edge_block(graph)
