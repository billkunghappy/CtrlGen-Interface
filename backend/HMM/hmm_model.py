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
        B_cache = torch.empty(len(suffix_tokens), hidden_states, device=device)
        y = torch.zeros(hidden_states, device=device)
        for t in range(len(suffix_tokens)-1, -1, -1):
            if t != len(suffix_tokens) - 1:
                y = matmul_a_logb(alpha_exp, y[:, None]).squeeze(-1)
            y = y + beta[:, suffix_tokens[t]]
            B_cache[t, :] = y

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

        C = torch.empty(max_tokens+1, num_states, hidden_states, device=device)
        C[0, :, :] = y
        for t in range(1, max_tokens+1):
            y = matmul_a_logb(EV_mask, y) # num_transitions * hidden_states
            y.nan_to_num_(neginf=-1e30)
            y = matmul_a_logb(VE_mask, T_weights + y) # num_states * hidden_states
            y.nan_to_num_(neginf=-1e30)
            y = matmul_loga_b(y, alpha_exp_t) # num_states * hidden_states
            C[t, :, :] = y

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
        C.nan_to_num_(neginf=-1e30)

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
        generation_offset, token_ranges, batch_size=8):

        device = self.alpha_exp.device
        eos_token_id = 2
        neginf_cuda = -1e10 * torch.ones(1, device=device)

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

        logits = torch.full((prefix_num, vocab_size), -1e10, device=device)

        # gather the list of indices that has at least one more token left before suffix
        generated_tokens = prefix_len - generation_offset
        selected_idx = [prefix_idx for prefix_idx, prefix in enumerate(prefixes)
            if token_ranges[prefix_idx][1] - generated_tokens > 0]
        selected_num = len(selected_idx)
        if len(selected_idx) > 0:
            for batch_idx in range(0, selected_num, batch_size):
                batch_size_ = min(batch_size, selected_num - batch_idx)
                selected_batch = selected_idx[batch_idx: batch_idx+batch_size_]

                A_batch = A[selected_batch] # batch_size_ * hidden_states

                prefixes_batch = [prefixes[i] for i in selected_batch]

                C_batch = []
                for prefix_idx in selected_batch:
                    min_tokens, max_tokens = token_ranges[prefix_idx]
                    remaining_tokens_max = max_tokens - generated_tokens
                    remaining_tokens_min = max(1, min_tokens - generated_tokens)
                    C_batch.append(C_cache[(remaining_tokens_min-1, remaining_tokens_max-1)])
                C_batch = torch.stack(C_batch, dim=0) # batch_size_ * num_states * hidden_states

                C = A_batch[:, None, :] + C_batch # batch_size_ * num_states * hidden_states

                C_shape = C.shape
                C = matmul_log(torch.flatten(C, start_dim=0, end_dim=1), beta) # (batch_size_ * num_states) * vocab_size
                C = C.view(C_shape[0], C_shape[1], -1) # batch_size_ * num_states * vocab_size

                mask = torch.stack([VE_mask[D_cache[prefix]] for prefix in prefixes_batch], dim=0) # prefix_mask, batch_size_ * num_transitions
                mask = mask[:, :, None] * EV_mask[None, :, :] # batch_size_ * num_transitions * num_states
                mask = torch.transpose(mask, 1, 2) # batch_size_ * num_states * num_transitions

                mask_shape = mask.shape
                mask = torch.matmul(torch.flatten(mask, start_dim=0, end_dim=1), T_mask) # (batch_size_ * num_states) * vocab_size
                mask = mask.view(mask_shape[0], mask_shape[1], -1) # batch_size_ * num_states * vocab_size
                mask = torch.nan_to_num(torch.log(mask), neginf=-1e30)

                logits_batch = logsumexp(C + mask, dim=1) # batch_size_ * vocab_size

                logits[selected_batch, :] = logits_batch

        # if current prefix already ends with part/none of the suffix; no hmm mini-batch here
        # TODO: potential optimization
        for prefix_idx, prefix in enumerate(prefixes):
            min_tokens, max_tokens = token_ranges[prefix_idx]
            offset_min = min_tokens + generation_offset
            offset_max = max_tokens + generation_offset
            offsets = ends_at(prefix, suffix,
                offset_min, D_cache, dfa_model)
            for offset in offsets:
                log_prob = logsumexp(A[prefix_idx] + B_cache[offset], dim=0)
                logits[prefix_idx, suffix[offset]] = torch.logaddexp(logits[prefix_idx, suffix[offset]], log_prob)


        # compute normalizing constant; no hmm mini-batch here
        logits_ = matmul_log(A, beta)

        # early termination if suffix does not end with eos
        # TODO: only to handle the case when Suffix == '', might not work in general
        if suffix[-1] != eos_token_id and generated_tokens > 0:
            for prefix_idx, prefix in enumerate(prefixes):
                if prefix[-len(suffix):] == suffix:
                    logits[prefix_idx, eos_token_id] = -neginf_cuda
                    logits_[prefix_idx, eos_token_id] = 0.0

        return logits, logits_
