import torch


def retrieve_past_key_values(past_key_values, i, generated_tokens):
    # retrieve the past key values
    new_past_key_values = []

    # TODO: retrieve the past key values
    # the kv cache is like:
    # past_key_values = [
    #     # layer 1
    #     [
    #         [key, value],
    #         [key, value],
    #         ...
    #     ],
    #     # layer 2
    #     [...],
    #     ...
    # ]
    # and the size of key/value is (batch_size, num_heads, padded_seq_len, head_dim)
    # you need to retrieve the past key values for the every sequence in the batch
    # the size of the retrieved key/value should be (num_heads, unpadded_seq_len, head_dim) with same structure
    # ==== start your code here ====
    
    for layer_kv in past_key_values:
        layer_past_kv = []
        key_i = layer_kv[0][i:i+1].squeeze(0)
        value_i = layer_kv[1][i:i+1].squeeze(0)
        key_i = key_i[:, :generated_tokens, :]
        value_i = value_i[:, :generated_tokens, :]
        layer_past_kv.append(key_i)
        layer_past_kv.append(value_i)
        new_past_key_values.append(layer_past_kv)
    
    # ==== end of your code ====
    return new_past_key_values


def prepare_inputs_for_prefill(seq_states, tokenizer, model):
    # generate the input ids
    for seq_state in seq_states:
        seq_state.input_ids = (
            tokenizer(seq_state.prompt, return_tensors="pt")
            .input_ids[0]
            .to(model.device)
        )
    input_ids = [seq_state.input_ids for seq_state in seq_states]

    # TODO: pad the input ids
    # pad the input ids with the eos_token to the max length in the batch
    # ==== start your code here ====
    max_length = max(len(ids) for ids in input_ids)
    eos_token_id = tokenizer.eos_token_id
    padded_input_ids = [torch.nn.functional.pad(ids, (0, max_length - ids.size(0)), value=eos_token_id) for ids in input_ids]
    padded_input_ids = torch.stack(padded_input_ids)
    # ==== end of your code ====


    # TODO: pad the attention mask
    # pad the attention mask with 0 to the max length in the batch
    # ==== start your code here ====
    attention_mask = []
    for seq_state in seq_states:
        attention_mask.append(
            (
            tokenizer(seq_state.prompt, return_tensors="pt")
            .attention_mask[0]
            .to(model.device)
            )
        )
    max_length = max(len(mask) for mask in attention_mask)
    padded_attention_mask = [torch.nn.functional.pad(mask, (0, max_length - mask.size(0)), value=0) for mask in attention_mask]
    attention_mask = torch.stack(padded_attention_mask)
    # ==== end of your code ====
    return padded_input_ids, attention_mask


def prepare_inputs_for_decode(seq_states):
    # cat the input ids
    input_ids = torch.stack([seq_state.input_ids for seq_state in seq_states])

    # TODO: pad the attention mask
    # pad the attention mask with 0 to the max length in the batch
    # attention mask shape: (batch_size, seq_len)
    # ==== start your code here ====
    attention_mask = None
    max_length = max(seq_state.input_ids.size(0) for seq_state in seq_states)
    attention_mask = [
        torch.nn.functional.pad(torch.ones(seq_state.input_ids.size(0)), (0, max_length - seq_state.input_ids.size(0)), value=0)
        for seq_state in seq_states
    ]
    attention_mask = torch.stack(attention_mask).long().to(input_ids.device)

    # ==== end of your code ====


    # TODO: pad past key values
    # pad the past key values with 0 to the max length in the batch
    # ==== start your code here ====
    past_key_values = [seq_state.past_key_values for seq_state in seq_states]
    num_layers = len(past_key_values[0])  # Number of layers in past_key_values
    padded_past_key_values = []
    padded_past_key_values = []
    num_layers = len(seq_states[0].past_key_values)
    for layer_idx in range(num_layers):
        layer_keys = []
        layer_values = []
        seq_lengths = [seq_state.past_key_values[layer_idx][0].size(1) for seq_state in seq_states]
        max_seq_length = max(seq_lengths)
        for seq_state in seq_states:
            key, value = seq_state.past_key_values[layer_idx]
            current_seq_length = key.size(1)

            # Pad keys and values to max_seq_length
            padded_key = torch.nn.functional.pad(
                key,
                (0, 0, 0, max_seq_length - current_seq_length),
                value=0
            )
            padded_value = torch.nn.functional.pad(
                value,
                (0, 0, 0, max_seq_length - current_seq_length),
                value=0
            )
            
            layer_keys.append(padded_key)
            layer_values.append(padded_value)
        
        # Stack keys and values for this layer
        # Shape: (batch_size, num_heads, max_seq_length, head_dim)
        stacked_keys = torch.stack(layer_keys, dim=0)
        stacked_values = torch.stack(layer_values, dim=0)
        
        padded_past_key_values.append((stacked_keys, stacked_values))
    # ==== end of your code ====
    return attention_mask, tuple(padded_past_key_values), input_ids


def embedding_only(seq_states, model, tokenizer):
    padded_input_ids, attention_mask = prepare_inputs_for_prefill(
        seq_states, tokenizer, model
    )

    # forward
    out = model.forward(
        padded_input_ids, attention_mask=attention_mask, output_hidden_states=True
    )

    # TODO: get the embedding
    # you need to get the embedding of the last layer's output (mean across sequence)
    # and set it to the embedding attribute of each seq_state
    # ==== start your code here ====
    last_hidden_state = out.hidden_states[-1]
    embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    for i, seq_state in enumerate(seq_states):
        seq_state.embedding = embeddings[i].detach().cpu().tolist()
    # ==== end of your code ====
    return seq_states


def prefill(seq_states, model, tokenizer):
    padded_input_ids, attention_mask = prepare_inputs_for_prefill(
        seq_states, tokenizer, model
    )
    # forward
    out = model.forward(padded_input_ids, attention_mask=attention_mask)
    # get the next input tokens
    past_key_values = out.past_key_values
    model_inputs = out.logits[:, -1].argmax(dim=-1).unsqueeze(1)
    decoded_tokens = tokenizer.batch_decode(model_inputs, skip_special_tokens=True)

    # TODO:update the seq states
    # including: generated_tokens(int), input_ids(torch.Tensor), has_prefilled(bool), decoded_tokens(str), past_key_values(list[tuple[torch.Tensor, torch.Tensor]])
    # ==== start your code here ====
    for i, seq_state in enumerate(seq_states):
        seq_state.generated_tokens = len(seq_states[i].input_ids) + len(decoded_tokens[i])
        seq_state.input_ids = model_inputs[i]
        seq_state.has_prefilled = True
        seq_state.decoded_tokens = decoded_tokens[i]
        seq_state.past_key_values = [(
            past_key_values[layer_idx][0][i],  # Select the key tensor for this sequence
            past_key_values[layer_idx][1][i]   # Select the value tensor for this sequence
        ) for layer_idx in range(len(past_key_values))]
    # ==== end of your code ====
    return seq_states


def decode(seq_states, model, tokenizer):
    # extend the attention mask
    attention_mask, past_key_values, input_ids = prepare_inputs_for_decode(seq_states)
    # forward
    out = model.forward(
        input_ids, attention_mask=attention_mask, past_key_values=past_key_values
    )
    # get the next input token
    model_inputs = out.logits[:, -1].argmax(dim=-1).unsqueeze(1)
    decoded_tokens = tokenizer.batch_decode(model_inputs, skip_special_tokens=True)
    # update the seq states
    # TODO:update the seq states
    # including: generated_tokens(int), input_ids(torch.Tensor), decoded_tokens(str), past_key_values(list[tuple[torch.Tensor, torch.Tensor]])
    # ==== start your code here ====
    for i, seq_state in enumerate(seq_states):
        seq_state.generated_tokens += len(decoded_tokens[i])
        seq_state.input_ids = model_inputs[i]
        seq_state.decoded_tokens = seq_state.decoded_tokens + decoded_tokens[i]
        seq_state.past_key_values = retrieve_past_key_values(out.past_key_values, i, seq_state.generated_tokens)
    # ==== end of your code ====
    return seq_states


def serve_step(model, tokenizer, seq_states):
    # prefill if any sequence not prefilled
    prefill_list = []
    embedding_only_list = []
    for seq_state in seq_states:
        if not seq_state.has_prefilled:
            prefill_list.append(seq_state)
        if seq_state.embedding_only:
            embedding_only_list.append(seq_state)

    if len(embedding_only_list) != 0:
        embedding_only(embedding_only_list, model, tokenizer)
        consumed_tokens = 0

    elif len(prefill_list) != 0:
        prefill(prefill_list, model, tokenizer)
        consumed_tokens = sum(seq_state.generated_tokens for seq_state in prefill_list)

    else:
        decode(seq_states, model, tokenizer)
        consumed_tokens = len(seq_states)

    return consumed_tokens
