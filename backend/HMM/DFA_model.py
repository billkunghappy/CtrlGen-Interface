import torch
from queue import Queue


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


def DFA_merge_states(A):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']

    G = edges2G(edges)
    duplicate_states = set([u for u in G if len(G[u]) == 1 and G[u][0] == u])

    state2parent = {}
    if len(duplicate_states) > 0:
        rep_accept, rep_dead = None, None
        for state in duplicate_states:
            if state in accept_states:
                if rep_accept is None:
                    rep_accept = state
                state2parent[state] = rep_accept
            else:
                if rep_dead is None:
                    rep_dead = state
                state2parent[state] = rep_dead

    edges_ = []
    for edge in edges:
        u, v, transition = edge
        u = u if u not in state2parent else state2parent[u]
        v = v if v not in state2parent else state2parent[v]
        edges_.append((u, v, transition))

    initial_state_ = initial_state if initial_state not in state2parent else state2parent[initial_state]

    accept_states_ = set()
    for state in accept_states:
        state = state if state not in state2parent else state2parent[state]
        accept_states_.add(state)

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
    while not Q.empty():
        u = Q.get()
        for v in G[u]:
            if v not in reachable_states:
                reachable_states.add(v)
                Q.put(v)

    edges_ = [edge for edge in edges
        if (edge[0] in reachable_states and edge[1] in reachable_states)]
    accept_states_ = [state for state in accept_states
        if state in reachable_states]

    return {
        'edges': edges_,
        'initial_state': initial_state,
        'accept_states': accept_states_
    }


def DFA_minimize(A):
    A = DFA_remove_unreachable_states(A)
    # A = DFA_merge_states(A)
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
                transition = EA[(ua, va)].intersection(EB[(ub, vb)])
                if len(transition) > 0:
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
        self.vocab_set = set([x for x in range(0, vocab_size)])


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
        candidate_tokens = set(pat)

        E = {}
        for u in range(0, len(pat)):
            for token in candidate_tokens:
                if token == pat[u]:
                    v = u + 1
                else:
                    v = 0 if u == 0 else compute_lps_i(pat, lps, lps[u-1], token)

                if (u, v) not in E:
                    E[(u, v)] = set()
                E[(u, v)].add(token)

            if (u, 0) not in E:
                E[(u, 0)] = set()
            E[(u, 0)].update(self.vocab_set.difference(candidate_tokens))

        E[(len(pat), len(pat))] = self.vocab_set

        edges = []
        for e, transition in E.items():
            if transition != []:
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


# Generated text must be end with . " ? or !
# class EndSentenceBuilder:
#     def __init__(self, tokenizer, vocab_size,
#             periods=['.','\"', '?', '!'], eos_token_id=2):

#         vocab_set = set([x for x in range(0, vocab_size)])
#         token_ids = [tokenizer.encode(f'\n{period}')[3] for period in set(periods)]
#         others_set = vocab_set.difference(set(token_ids))

#         edges = []
#         for idx, token_id in enumerate(token_ids):
#             edges.append((-1, idx, set([token_id])))

#         for idx, _ in enumerate(token_ids):
#             edges.append((idx, -2, set([eos_token_id])))

#         for idx, _ in enumerate(token_ids):
#             for jdx, token_id in enumerate(token_ids):
#                 edges.append((idx, jdx, set([token_id])))

#         for idx, _ in enumerate(token_ids):
#             edges.append((idx, -1, others_set))
#         edges.append((-1, -1, others_set))

#         edges.append((-2, -2, vocab_set))

#         initial_state = -1
#         accept_states = set([-2])

#         self.dfa_graph = {
#             'edges': edges,
#             'initial_state': initial_state,
#             'accept_states': accept_states,
#         }


#     def build(self):
#         return self.dfa_graph

class EndSentenceBuilder:
    def __init__(self, tokenizer, vocab_size,
            periods=['.','\"', '?', '!'], eos_token_id=2):

        vocab_set = set([x for x in range(0, vocab_size)])
        token_ids = [tokenizer.encode(f'\n{period}')[3] for period in set(periods)]
        others_set = vocab_set.difference(set(token_ids))

        edges = [
            (0, 1, set(token_ids)),
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

        vocab_set = set([x for x in range(0, vocab_size)])

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

        vocab_set = set([x for x in range(0, vocab_size)])
        eos, others = set([eos_token_id]), vocab_set.difference(set([eos_token_id]))

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
    def __init__(self, tokenizer, vocab_size, sep=[' ', '\n', ',', '.', ':', ';', '\"', '\'', '/']):
        vocab00_list, vocab01_list, vocab10_list, vocab11_list = [], [], [], []
        for token_id in range(3, vocab_size):
            token = tokenizer.decode([13, token_id])[1:]
            if token[0] in sep:
                if any([c.isalpha() or c.isdigit() for c in token]):
                    vocab11_list.append(token_id)
                else:
                    vocab10_list.append(token_id)
            else:
                if any([c.isalpha() or c.isdigit() for c in token]):
                    vocab01_list.append(token_id)
                else:
                    vocab00_list.append(token_id)
        vocab00_list.extend([0,1,2])

        vocab0x = vocab00_list + vocab01_list
        vocabx0 = vocab00_list + vocab10_list
        vocabx1 = vocab01_list + vocab11_list
        vocab10 = vocab10_list
        vocab11 = vocab11_list

        self.vocab0x = set(vocab0x)
        self.vocabx0 = set(vocabx0)
        self.vocabx1 = set(vocabx1)
        self.vocab10 = set(vocab10)
        self.vocab11 = set(vocab11)
        self.vocab_set = set([token for token in range(0, vocab_size)])


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

        initial_state = (0, 0)
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
        VE_mask = torch.zeros(state_cnt, edge_cnt, device=device)
        EV_mask = torch.zeros(edge_cnt, state_cnt, device=device)
        T_mask = torch.zeros(edge_cnt, vocab_size, device=device)
        for e in edges:
            u, v, transition = e    # transition should be a set of tokens
            u_idx, v_idx = state2idx[u], state2idx[v]
            edge_idx = edge2idx[(u_idx, v_idx)]
            VE_mask[u_idx, edge_idx] = 1.0
            EV_mask[edge_idx, v_idx] = 1.0
            T_mask[edge_idx, list(transition)] = 1.0

            if u_idx not in G:
                G[u_idx] = []
            G[u_idx].append((v_idx, transition))

        self.G = G
        self.VE_mask = VE_mask
        self.EV_mask = EV_mask
        self.T_mask = T_mask
        self.num_states = state_cnt
        self.initial_state = state2idx[initial_state]
        self.accept_states = set([state2idx[x] for x in accept_states])


    def next_state(self, state, token):
        for e in self.G[state]:
            v, transition_set = e
            if token in transition_set:
                return v
        print('ERROR: no valid transition!')
        exit(1)


    def is_accept(self, state):
        return state in self.accept_states


# class KeyphraseBuilder:
#     def __init__(self, tokenizer, vocab_size):
#         self.tokenizer = tokenizer
#         self.vocab_set = set([x for x in range(0, vocab_size)])
#         self.vocab_size = vocab_size


#     def build(self, keyphrases):
#         tokenizer = self.tokenizer
#         vocab_size = self.vocab_size
#         keyphrase_ids = [tuple(tokenizer.encode(x)[1:]) for x in keyphrases]

#         def gen_states(A):
#             if len(A) == 1:
#                 return [(x,) for x in range(0, A[0]+1)]
#             res_ = gen_states(A[1:])
#             res = [(x,) + y for x in range(0, A[0]+1) for y in res_]
#             return res

#         def next_state(state, token):
#             new_state = []
#             for x, keyphrase in zip(state, keyphrase_ids):
#                 if x < len(keyphrase):
#                     new_state.append(x + 1 if keyphrase[x] == token else 0)
#                 else:
#                     new_state.append(x)
#             return tuple(new_state)

#         states = gen_states([len(x) for x in keyphrase_ids])

#         E = {}
#         for u in states:
#             for v in states:
#                 E[(u, v)] = []

#         for u in states:
#             candidate_tokens = set()
#             for x, phrase in zip(u, keyphrase_ids):
#                 if x < len(phrase):
#                     candidate_tokens.add(phrase[x])
#             for token in candidate_tokens:
#                 v = next_state(u, token)
#                 E[(u, v)].append(token)

#             v = ()
#             for x, phrase in zip(u, keyphrase_ids):
#                 if x < len(phrase):
#                     v = v + (0,)
#                 else:
#                     v = v + (x,)

#             E[(u, v)].extend(list(self.vocab_set.difference(candidate_tokens)))

#         edges = []
#         for e, transition in E.items():
#             if transition != []:
#                 u, v = e
#                 edges.append((u, v, transition))

#         initial_state = tuple([0] * len(keyphrase_ids))
#         accept_states = [tuple([len(x) for x in keyphrase_ids])]

#         return DFA_minimize({
#             'edges': edges,
#             'initial_state': initial_state,
#             'accept_states': accept_states,
#         })