import numpy as np
import ctrlg

class TrivialBuilder:
    def __init__(self, tokenizer, vocab_size,
            eos_token_id=2):

        vocab_set = np.ones((vocab_size,), dtype=bool) # set([x for x in range(0, vocab_size)])

        self.dfa_graph = {
            'edges': [(0, 1, vocab_set),
                        (1, 0, vocab_set)],
            'initial_state': 0,
            'accept_states': set([0, 1]),
        }


    def build(self):
        return self.dfa_graph
class EndSentenceBuilder:
    def __init__(self, tokenizer, vocab_size,
            periods=['.'], eos_token_id=2):

        vocab_set = np.ones((vocab_size,), dtype=bool) # ~bitarray(vocab_size)
        # token_ids = [tokenizer.encode(f'\n{period}')[3] for period in set(periods)]
        token_ids = ctrlg.set2npset([29889], vocab_size)
        others_set = ~token_ids

        edges = [
            (0, 1, token_ids),
            (0, 0, others_set),
            (1, 1, vocab_set)
        ]

        initial_state = 0
        accept_states = set([0])

        self.dfa_graph = {
            'edges': edges,
            'initial_state': initial_state,
            'accept_states': accept_states,
        }


    def build(self):
        return self.dfa_graph
