import torch


def span_masking(inputs, mask_embedding, mask_probability=0.08, span_length=10):
    B, L, D = inputs.size()
    inputs = inputs.clone()
    mask = torch.rand(B, L) > mask_probability
    span_mask = mask.clone()
    for i in range(B):
        for j in range(L):
            if mask[i, j] == False:
                span_mask[i, j : j + span_length] = False
    inputs[~span_mask] = mask_embedding
    return inputs


if __name__ == "__main__":
    inputs = torch.ones(1, 100, 3)
    mask_embedding = torch.zeros(3)
    masked = span_masking(inputs,mask_embedding)
    print("inputs", inputs)
    print("masked", masked)
