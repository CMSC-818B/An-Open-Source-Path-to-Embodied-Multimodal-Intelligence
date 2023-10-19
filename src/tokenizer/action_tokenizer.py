import torchrl


class ActionTokenizer:
  def __init__(self,
               action_spec: dict,
               vocab_size: int,
               action_order=None):
    self.action_spec = action_spec
    self.vocab_size = vocab_size
    if action_order is None:
      self.action_order = self.action_spec.keys()
    else:
      for action in action_order:
        if action not in self.action_spec.keys():
          raise ValueError(f"actions: {action} not found in action_spec: {action_spec.keys()}")
        assert action in self.action_spec.keys()
      self.action_order = action_order
  
  def tokenize(self, actions):
    """Tokenizes an action."""
    action_tokens = []
    for k in self.action_order:
      a = actions[k]  # a is [batch, actions_size]
      spec = self.action_spec[k]
      a = torch.clip(a, spec.space.low, spec.space.high)
      # Normalize the action [batch, actions_size]
      token = (a - spec.space.low) / (spec.space.high - spec.space.low)
      # Bucket and discretize the action to vocab_size, [batch, actions_size]
      token = (token * (self.vocab_size -1))
      token = token.type(torch.int32)
      action_tokens.append(token)
    # Append all actions, [batch, all_actions_size]
    action_tokens = torch.cat(action_tokens, axis=-1)
    return action_tokens

  def detokenize(self, action_tokens):
    """Detokenizes an action."""
    action = {}
    token_index = 0
    for k in self.action_order:
      spec = self.action_spec[k]
      action_dim = spec.shape[0]
      actions = []
      for _ in range(action_dim):
        spec_index = token_index % action_dim
        a = action_tokens[..., token_index:token_index + 1]
        a = a.type(torch.float32)
        a = a / (self.vocab_size - 1)
        a = (a * (spec.space.high[spec_index] - spec.space.low[spec_index])) + spec.space.low[spec_index]
        actions.append(a)
        token_index += 1
      action[k] = torch.cat(actions, axis=-1)
    return action
