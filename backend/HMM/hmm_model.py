import torch
import torch.nn as nn


torch.set_float32_matmul_precision('high')

@torch.compile
def logsumexp(A, dim):
    return torch.logsumexp(A, dim)


@torch.compile
def matmul_log(A, B):
    bd = len(B.shape) - 2
    A_max = torch.amax(A, dim=-1, keepdim=True)
    B_max = torch.amax(B, dim=bd, keepdim=True)
    A = A - A_max
    B = B - B_max
    A.exp_()
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max + B_max)

    return C


@torch.compile
def matmul_loga_b(A, B):
    A_max = torch.amax(A, dim=-1, keepdim=True)
    A = A - A_max
    A.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max)

    return C


@torch.compile
def matmul_a_logb(A, B):
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B = B - B_max
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(B_max)

    return C


def ends_at(prefix, suffix,
    offset_min, D_cache, dfa_model):
    ans = []
    for s in range(0, len(suffix)):
        offset = len(prefix) - s
        if offset < offset_min:
            break
        state = D_cache[tuple(prefix[:-s])] if s != 0 else D_cache[tuple(prefix)]
        if dfa_model.is_accept(state):
            if s == 0 or suffix[:s] == prefix[-s:]:
                ans.append(s)
    return ans


class HMM(nn.Module):
    def __init__(self, weights_file):
        super().__init__()

        assert(weights_file[-2:] == 'th')

        d = torch.load(weights_file)
        alpha, beta, gamma = d['alpha'], d['beta'], d['gamma']

        alpha_exp = torch.softmax(alpha, dim=1)
        beta = torch.log_softmax(beta, dim=1)
        gamma = torch.log_softmax(gamma, dim=0)

        hidden_states, _ = beta.shape

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=False)
        self.beta = nn.Parameter(beta, requires_grad=False)
        self.gamma = nn.Parameter(gamma, requires_grad=False)

        self.dfa_model = None

        self.cache = {}


    def initialize_cache(self, prefix_tokens, suffix_tokens, token_ranges, dfa_model, ignore_prefix=False):

        device = self.alpha_exp.device
        hidden_states, vocab_size = self.beta.shape
        num_states = dfa_model.num_states
        neginf_cuda = -1e30 * torch.ones(1, device=device)

        alpha_exp, beta, gamma = self.alpha_exp, self.beta, self.gamma
        alpha_exp_t = torch.transpose(alpha_exp, 0, 1)

        max_tokens = max([x[1] for x in token_ranges])

        # precompute renages for C_cache
        max_tokens = max([x[1] for x in token_ranges])
        ranges = set()
        for token_range in token_ranges:
            min_tokens_, max_tokens_ = token_range
            for i in range(min_tokens_, -1, -1):
                ranges.add((i, i + max_tokens_ - min_tokens_))

            for i in range(max_tokens_ - min_tokens_ - 1, -1, -1):
                ranges.add((0, i))

        ranges = list(ranges)
        range_mask = torch.zeros(len(ranges), max_tokens+1, device=device)
        for idx, r in enumerate(ranges):
            range_mask[idx, torch.arange(r[0], r[1]+1)] = 1.0

        # initialize cache A
        A_cache = {}
        y = gamma.clone()
        if not ignore_prefix:
            for t in range(0, len(prefix_tokens)):
                y = y + beta[:, prefix_tokens[t]]
                y = matmul_loga_b(y[None, :], alpha_exp).squeeze(0)
        A_cache[tuple(prefix_tokens)] = y

        # initialize cache B
        B_cache = {}
        y = torch.zeros(hidden_states, device=device)
        for t in range(len(suffix_tokens)-1, -1, -1):
            if t != len(suffix_tokens) - 1:
                y = matmul_a_logb(alpha_exp, y[:, None]).squeeze(-1)
            y = y + beta[:, suffix_tokens[t]]
            B_cache[-t] = y

        # compute T_weights
        T_mask = dfa_model.T_mask
        VE_mask = dfa_model.VE_mask
        EV_mask = dfa_model.EV_mask
        T_weights = matmul_a_logb(T_mask, torch.transpose(beta, 0, 1)) # num_transitions * hidden_states
        T_weights.nan_to_num_(neginf=-1e30)

        # initialize cache C
        C_cache = {}

        y_ = torch.full((num_states, hidden_states), -1e30, device=device)
        y_[list(dfa_model.accept_states), :] = y
        y = matmul_loga_b(y_, alpha_exp_t) # num_states * hidden_states

        # C = [y]
        C = torch.empty(max_tokens+1, num_states, hidden_states, device=device)
        C[0, :, :] = y
        for t in range(1, max_tokens+1):
            y = matmul_a_logb(EV_mask, y) # num_transitions * hidden_states
            y.nan_to_num_(neginf=-1e30)
            y = matmul_a_logb(VE_mask, T_weights + y) # num_states * hidden_states
            y.nan_to_num_(neginf=-1e30)
            y = matmul_loga_b(y, alpha_exp_t) # num_states * hidden_states
            C[t, :, :] = y
            # C.append(y)
        # C = torch.stack(C, dim=0) # (max_tokens+1) * num_states * hidden_states

        ranges = set()
        for token_range in token_ranges:
            min_tokens_, max_tokens_ = token_range
            for i in range(min_tokens_, -1, -1):
                ranges.add((i, i + max_tokens_ - min_tokens_))

            for i in range(max_tokens_ - min_tokens_ - 1, -1, -1):
                ranges.add((0, i))

        ranges = list(ranges)
        range_mask = torch.zeros(len(ranges), max_tokens+1, device=device)
        for idx, r in enumerate(ranges):
            range_mask[idx, torch.arange(r[0], r[1]+1)] = 1.0

        C_shape = C.shape
        C = matmul_a_logb(range_mask, torch.flatten(C, start_dim=1, end_dim=2)) # num_ranges * (num_states * hidden_states)
        C = C.view(-1, C_shape[1], C_shape[2])
        # C = torch.maximum(C, neginf_cuda)
        C.nan_to_num_(neginf=-1e30)
        # C = torch.unbind(C, dim=0)

        for idx, r in enumerate(ranges):
            C_cache[r] = C[idx]

        # initialize cache D
        D_cache = {tuple(prefix_tokens): dfa_model.initial_state}

        self.cache['A'] = A_cache
        self.cache['B'] = B_cache
        self.cache['C'] = C_cache
        self.cache['D'] = D_cache
        self.cache['VE_mask'] = VE_mask
        self.cache['EV_mask'] = EV_mask
        self.cache['T_mask'] = T_mask
        self.dfa_model = dfa_model


    def update_A(self, prefixes):
        A_cache = self.cache['A']

        A = torch.stack([A_cache[prefix[:-1]] for prefix in prefixes], dim=0) # len(prefixes) * hidden_states
        log_probs = torch.stack([self.beta[:, prefix[-1]] for prefix in prefixes], dim=0) # len(prefixes) * hidden_states
        A += log_probs
        A = matmul_loga_b(A, self.alpha_exp)

        for i, prefix in enumerate(prefixes):
            A_cache[prefix] = A[i]

        return A


    def update_D(self, prefixes):
        D_cache = self.cache['D']
        for prefix in prefixes:
            next_state = self.dfa_model.next_state(D_cache[prefix[:-1]], prefix[-1])
            D_cache[prefix] = next_state


    # compute logits for next_token:
    def compute_logits(self, prefixes, suffix,
        generation_offset, min_tokens, max_tokens, batch_size=8):

        device = self.alpha_exp.device
        neginf_cuda = -1e30 * torch.ones(1, device=device)

        prefix_num, prefix_len = len(prefixes), len(prefixes[0])

        dfa_model = self.dfa_model
        VE_mask, EV_mask, T_mask = self.cache['VE_mask'], self.cache['EV_mask'], self.cache['T_mask']
        A_cache, B_cache, C_cache, D_cache = self.cache['A'], self.cache['B'], self.cache['C'], self.cache['D']
        alpha_exp, beta, gamma = self.alpha_exp, self.beta, self.gamma
        hidden_states, vocab_size = self.beta.shape

        # update prefix hidden states
        if prefix_len > generation_offset:
            A = self.update_A(prefixes)
            self.update_D(prefixes)
        else:
            A = torch.stack([A_cache[prefix] for prefix in prefixes], dim=0) # prefix_num * hidden_states

        logits = torch.full((prefix_num, vocab_size), -1e30, device=device)

        # generate suffix at least one token after current prefix
        generated_tokens = prefix_len - generation_offset
        remaining_tokens_max = max_tokens - generated_tokens
        remaining_tokens_min = max(1, min_tokens - generated_tokens)
        if remaining_tokens_max > 0:
            logits = []
            for batch_idx in range(0, prefix_num, batch_size):
                batch_size_ = min(batch_size, prefix_num - batch_idx)
                A_batch = A[batch_idx: batch_idx+batch_size_]
                prefixes_batch = prefixes[batch_idx: batch_idx+batch_size_]

                C = C_cache[(remaining_tokens_min-1, remaining_tokens_max-1)] # num_states * hidden_states
                C = A_batch[:, None, :] + C[None, :, :] # prefix_num * num_states * hidden_states

                C_shape = C.shape
                C = matmul_log(torch.flatten(C, start_dim=0, end_dim=1), beta) # (prefix_num * num_states) * vocab_size
                C = C.view(C_shape[0], C_shape[1], -1) # prefix_num * num_states * vocab_size

                mask = torch.stack([VE_mask[D_cache[prefix]] for prefix in prefixes_batch], dim=0) # prefix_mask, prefix_num * num_transitions
                mask = mask[:, :, None] * EV_mask[None, :, :] # prefix_num * num_transitions * num_states
                mask = torch.transpose(mask, 1, 2) # prefix_num * num_states * num_transitions

                mask_shape = mask.shape
                mask = torch.matmul(torch.flatten(mask, start_dim=0, end_dim=1), T_mask) # (prefix_num * num_states) * vocab_size
                mask = mask.view(mask_shape[0], mask_shape[1], -1) # prefix_num * num_states * vocab_size
                mask = torch.nan_to_num(torch.log(mask), neginf=-1e30)

                logits_ = logsumexp(C + mask, dim=1) # prefix_num * vocab_size

                logits.append(logits_)

            logits = torch.cat(logits, dim=0)

        # if current prefix already ends with part/none of the suffix
        offset_min = min_tokens + generation_offset
        offset_max = max_tokens + generation_offset
        for prefix_idx, prefix in enumerate(prefixes):
            offsets = ends_at(prefix, suffix,
                offset_min, D_cache, dfa_model)
            for offset in offsets:
                log_prob = logsumexp(A[prefix_idx] + B_cache[-offset], dim=0)
                logits[prefix_idx, suffix[offset]] = torch.logaddexp(logits[prefix_idx, suffix[offset]], log_prob)

        # compute normalizing constant
        logits_ = []
        for batch_idx in range(0, prefix_num, batch_size):
            batch_size_ = min(batch_size, prefix_num - batch_idx)
            logits_.append(matmul_log(A[batch_idx:batch_idx+batch_size_], beta))
        logits_ = torch.cat(logits_, dim=0) # prefix_num * vocab_size

        return logits, logits_
