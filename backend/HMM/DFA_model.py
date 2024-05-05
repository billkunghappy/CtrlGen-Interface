import torch
from queue import Queue
import numpy as np


def set2npset(A, n):
    res = np.zeros((n,), dtype=bool)
    for x in A:
        res[x] = 1
    return res


def edges2G(edges):
    G = {}
    for edge in edges:
        u, v, transition = edge
        if u not in G:
            G[u] = []
        if len(transition) > 0:
            G[u].append(v)
    return G


def edges2states(edges):
    states = set()
    for edge in edges:
        u, v, _ = edge
        states.add(u)
        states.add(v)
    return states


def edges2dict(edges):
    res = {}
    for edge in edges:
        u, v, transition = edge
        res[(u, v)] = transition
    return res


def DFA_merge_undistinguishable_states(A):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']

    G = edges2G(edges)
    duplicate_states = set([u for u in G if len(G[u]) == 1 and G[u][0] == u])

    #TODO

    return {
        'edges': edges_,
        'initial_state': initial_state_,
        'accept_states': accept_states_,
    }


def DFA_remove_unreachable_states(A):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']

    G = edges2G(edges)

    reachable_states = set()
    Q = Queue()
    Q.put(initial_state)
    reachable_states.add(initial_state)
    while not Q.empty():
        u = Q.get()
        for v in G[u]:
            if v not in reachable_states:
                reachable_states.add(v)
                Q.put(v)

    edges_ = [edge for edge in edges
        if (edge[0] in reachable_states and edge[1] in reachable_states)]
    accept_states_ = set([state for state in accept_states
        if state in reachable_states])

    return {
        'edges': edges_,
        'initial_state': initial_state,
        'accept_states': accept_states_
    }


def DFA_minimize(A):
    A = DFA_remove_unreachable_states(A)
    # A = DFA_merge_undistinguishable_states(A)
    return A


def DFA_size(A):
    edge_cnt = len(A['edges'])
    states = set()
    for edge in A['edges']:
        states.add(edge[0])
        states.add(edge[1])
    state_cnt = len(states)

    return state_cnt, edge_cnt


def DFA_negate(A):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']

    all_states = set()
    for edge in edges:
        u, v, _ = edge
        all_states.add(u)
        all_states.add(v)

    accept_states_ = all_states.difference(accept_states)

    return {
        'edges': edges,
        'initial_state': initial_state,
        'accept_states': accept_states_
    }


def DFA_prod_binary(A, B, mode='intersection'):
    states_A = edges2states(A['edges'])
    states_B = edges2states(B['edges'])
    states_AB = [(ua, ub) for ua in states_A for ub in states_B]

    EA = edges2dict(A['edges'])
    EB = edges2dict(B['edges'])
    edges_AB = []
    for u in states_AB:
        for v in states_AB:
            ua, ub = u
            va, vb = v
            if (ua, va) in EA and (ub, vb) in EB:
                transition = EA[(ua, va)] & EB[(ub, vb)]
                if transition.any():
                    edges_AB.append((u, v, transition))

    initial_state_AB = (A['initial_state'], B['initial_state'])
    if mode == 'intersection':
        accept_states_AB = set([u for u in states_AB
            if u[0] in A['accept_states'] and u[1] in B['accept_states']])
    if mode == 'union':
        accept_states_AB = set([u for u in states_AB
            if u[0] in A['accept_states'] or u[1] in B['accept_states']])

    return DFA_minimize({
        'edges': edges_AB,
        'initial_state': initial_state_AB,
        'accept_states': accept_states_AB,
    })


def DFA_prod(dfa_graphs, mode='intersection'):
    if dfa_graphs == []:
        return []
    if len(dfa_graphs) == 1:
        return DFA_minimize(dfa_graphs[0])
    return DFA_prod_binary(dfa_graphs[0], DFA_prod(dfa_graphs[1:], mode=mode), mode=mode)


class PatternBuilder:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


    def build(self, pat):

        def compute_lps_i(pattern, lps, l, x):
            if x == pattern[l]:
                l += 1
            else:
                while l != 0:
                    l = lps[l - 1]
                    if x == pattern[l]:
                        l += 1
                        break
            return l

        def compute_lps(pattern):
            m = len(pattern)
            lps = [0] * m
            l = 0
            for i in range(1, m):
                l = compute_lps_i(pattern, lps, l, pattern[i])
                lps[i] = l
            return tuple(lps)

        lps = compute_lps(pat)

        pat_tokens_set = set(pat)
        candidate_tokens = set2npset(pat_tokens_set, self.vocab_size)

        E = {}
        for u in range(0, len(pat)):
            for token in pat_tokens_set:
                if token == pat[u]:
                    v = u + 1
                else:
                    v = 0 if u == 0 else compute_lps_i(pat, lps, lps[u-1], token)

                if (u, v) not in E:
                    E[(u, v)] = np.zeros((self.vocab_size,), dtype=bool) # bitarray(self.vocab_size)
                E[(u, v)][token] = 1

            if (u, 0) not in E:
                E[(u, 0)] = np.zeros((self.vocab_size,), dtype=bool) # bitarray(self.vocab_size)
            E[(u, 0)] |= ~candidate_tokens

        E[(len(pat), len(pat))] = np.ones((self.vocab_size,), dtype=bool) # ~bitarray(self.vocab_size)

        edges = []
        for e, transition in E.items():
            if transition.any():
                u, v = e
                edges.append((u, v, transition))

        initial_state = 0
        accept_states = set([len(pat)])

        return {
            'edges': edges,
            'initial_state': initial_state,
            'accept_states': accept_states
        }


class KeyphraseBuilder:
    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.pattern_builder = PatternBuilder(vocab_size)


    def build(self, keyphrases):
        tokenizer = self.tokenizer
        patterns = [tuple(tokenizer.encode(x)[1:]) for x in keyphrases]

        dfa_graphs = [self.pattern_builder.build(pattern) for pattern in patterns]
        dfa_graph = DFA_prod(dfa_graphs, mode='intersection')

        return dfa_graph


class BanphraseBuilder:
    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.pattern_builder = PatternBuilder(vocab_size)


    def build(self, banphrases):
        tokenizer = self.tokenizer
        patterns = [tuple(tokenizer.encode(x)[1:]) for x in banphrases]

        dfa_graphs = [self.pattern_builder.build(pattern) for pattern in patterns]
        dfa_graph = DFA_negate(DFA_prod(dfa_graphs, mode='union'))

        return dfa_graph


class EndSentenceBuilder:
    def __init__(self, tokenizer, vocab_size,
            periods=['.'], eos_token_id=2):

        vocab_set = np.ones((self.vocab_size,), dtype=bool) # ~bitarray(vocab_size)
        # token_ids = [tokenizer.encode(f'\n{period}')[3] for period in set(periods)]
        token_ids = set2npset([29889], vocab_size)
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


# A placeholder DFA that enforce no constraints
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


# EOS token must be followed by EOS token
class EOSBuilder:
    def __init__(self, tokenizer, vocab_size,
            eos_token_id=2):

        vocab_set = np.ones((self.vocab_size,), dtype=bool) # ~bitarray(vocab_size)
        eos = set2npset([eos_token_id], vocab_size)
        others = ~eos

        self.dfa_graph = {
            'edges': [(0, 1, eos),
                    (0, 0, others),
                    (1, 1, eos),
                    (1, 2, others),
                    (2, 2, vocab_set)],
            'initial_state': 0,
            'accept_states': set([0, 1]),
        }


    def build(self):
        return self.dfa_graph


class WordCountBuilder:
    def __init__(self, tokenizer, vocab_size, sep=[' ', '\n', ',', '.', ':', ';', '\"', '/']):
        vocab00, vocab01, vocab10, vocab11 = [np.zeros((vocab_size,), dtype=bool) for _ in range(0, 4)]
        for token_id in range(3, vocab_size):
            token = tokenizer.decode([13, token_id])[1:]
            if token[0] in sep:
                if any([c.isalpha() or c.isdigit() for c in token]):
                    vocab11[token_id] = 1 # vocab11_list.append(token_id)
                else:
                    vocab10[token_id] = 1 # vocab10_list.append(token_id)
            else:
                if any([c.isalpha() or c.isdigit() for c in token]):
                    vocab01[token_id] = 1 # vocab01_list.append(token_id)
                else:
                    vocab00[token_id] = 1 # vocab00_list.append(token_id)
        vocab00[:3] = 1 # vocab00_list.extend([0,1,2])

        self.vocab0x = vocab00 | vocab01
        self.vocabx0 = vocab00 | vocab10
        self.vocabx1 = vocab01 | vocab11        
        self.vocab10 = vocab10
        self.vocab11 = vocab11
        self.vocab_set = np.ones((vocab_size,), dtype=bool)


    def build(self, min_word_count, max_word_count):
        states = []
        states.extend([(k, s) for k in range(0, max_word_count+1) for s in range(0, 2)])
        states.append((max_word_count+1, 0))

        E = {}
        for u in states:
            k, s = u
            if k <= max_word_count:
                if s == 0:
                    E[(u, u)] = self.vocab0x
                    E[(u, (k, 1))] = self.vocab10
                    E[(u, (k+1, 0))] = self.vocab11
                if s == 1:
                    E[(u, u)] = self.vocabx0
                    E[(u, (k+1, 0))] = self.vocabx1
            else:
                E[(u, u)] = self.vocab_set

        edges = []
        for e, transition in E.items():
            u, v = e
            edges.append((u, v, transition))

        initial_state = (0, 1)
        accept_states = [(k, s) for k in range(min_word_count, max_word_count+1) for s in range(0, 2)]

        return {
            'edges': edges,
            'initial_state': initial_state,
            'accept_states': accept_states,
        }


class DFAModel:
    def __init__(self, dfa_graph, vocab_size=32000, device='cuda'):

        edges = dfa_graph['edges']
        initial_state = dfa_graph['initial_state']
        accept_states = dfa_graph['accept_states']

        state_cnt, edge_cnt = 0, 0
        state2idx, edge2idx = {}, {}

        # pre-process dfa_graph
        for e in edges:
            u, v, _ = e
            for x in [u, v]:
                if x not in state2idx:
                    state2idx[x] = state_cnt
                    state_cnt += 1
            u_idx, v_idx = state2idx[u], state2idx[v]
            if (u_idx, v_idx) not in edge2idx:
                edge2idx[(u_idx, v_idx)] = edge_cnt
                edge_cnt += 1
            else:
                print('ERROR: duplicate edge!')
                exit(1)

        G = {}
        VE_mask = torch.zeros(state_cnt, edge_cnt)
        EV_mask = torch.zeros(edge_cnt, state_cnt)
        T_mask = torch.zeros(edge_cnt, vocab_size)
        E2Src = torch.tensor([0] * edge_cnt)
        E2Dst = torch.tensor([0] * edge_cnt)
        for e in edges:
            u, v, transition = e    # transition should be a bitset of tokens
            u_idx, v_idx = state2idx[u], state2idx[v]
            edge_idx = edge2idx[(u_idx, v_idx)]
            VE_mask[u_idx, edge_idx] = 1.0
            EV_mask[edge_idx, v_idx] = 1.0
            T_mask[edge_idx, torch.from_numpy(transition)] = 1.0
            E2Src[edge_idx] = u_idx
            E2Dst[edge_idx] = v_idx

            if u_idx not in G:
                G[u_idx] = []
            G[u_idx].append((v_idx, transition))

        self.G = G
        self.VE_mask = VE_mask.to(device)
        self.EV_mask = EV_mask.to(device)
        self.T_mask = T_mask.to(device)
        self.E2Src = E2Src.to(device)
        self.E2Dst = E2Dst.to(device)
        self.num_states = state_cnt
        self.initial_state = state2idx[initial_state]
        self.accept_states = set([state2idx[x] for x in accept_states])


    def next_state(self, state, token):
        for e in self.G[state]:
            v, transition_set = e
            if transition_set[token] == 1:
                return v
        print('ERROR: no valid transition!')
        exit(1)


    def is_accept(self, state):
        return state in self.accept_states