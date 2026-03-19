import torch
import transformers
import numpy as np
IGNORE_INDEX = -100

def mask_non_assistant_labels(input_ids, tokenizer, ignore_index=IGNORE_INDEX):
    """
    Args:
        input_ids: torch.LongTensor, shape (B, T), tokenized conversations
        tokenizer: HuggingFace tokenizer
        ignore_index: int, value to use for masked tokens
    Returns:
        labels: torch.LongTensor, same shape as input_ids
    """
    
    # Special tokens - handle different tokenizer types
    try:
        # Try to get im_start_id and im_end_id directly if available
        start_id = getattr(tokenizer, 'im_start_id', None)
        end_id = getattr(tokenizer, 'im_end_id', None)
    except:
        start_id = None
        end_id = None
    
    # Fallback to convert_tokens_to_ids if direct attributes not available
    if start_id is None:
        start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if end_id is None:
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # Handle endoftext token
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    # Tokenize assistant role pattern
    assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
    
    # Initialize labels with ignore_index
    labels = torch.full_like(input_ids, ignore_index)

    for b in range(input_ids.size(0)):
        ids = input_ids[b]
        pos = 0
        
        while pos < len(ids):
            # Skip padding tokens
            if ids[pos] == eot_id:
                break
                
            if ids[pos] == start_id:
                # Found <|im_start|>, now check if followed by assistant role
                role_start = pos + 1
                
                # Check if the next tokens match "assistant\n"
                if role_start + len(assistant_tokens) <= len(ids):
                    role_tokens = ids[role_start:role_start + len(assistant_tokens)]
                    is_assistant = torch.equal(role_tokens, torch.tensor(assistant_tokens, device=ids.device))
                    
                    # Find the matching <|im_end|>
                    search_start = role_start + len(assistant_tokens)
                    end_positions = (ids[search_start:] == end_id).nonzero(as_tuple=True)[0]
                    
                    if len(end_positions) > 0:
                        # Found matching <|im_end|>
                        end_pos = end_positions[0].item() + search_start
                        
                        if is_assistant:
                            # This is an assistant round - keep tokens from after role until and including <|im_end|>
                            content_start = role_start + len(assistant_tokens)
                            
                            # Check if there's a newline after <|im_end|> and include it too
                            newline_end = end_pos + 1
                            if newline_end < len(ids):
                                # Check if the next token is a newline
                                newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]
                                if ids[newline_end] == newline_token:
                                    newline_end += 1
                            
                            labels[b, content_start:newline_end] = ids[content_start:newline_end]
                        
                        # Move to after this conversation block (including potential newline)
                        next_pos = end_pos + 1
                        if next_pos < len(ids):
                            newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]
                            if ids[next_pos] == newline_token:
                                next_pos += 1
                        pos = next_pos
                    else:
                        # No matching <|im_end|> found, skip this token
                        pos += 1
                else:
                    # Not enough tokens left to match assistant pattern
                    pos += 1
            else:
                pos += 1

    return labels

def count_trailing_mask(x, mask_value=-100):
    # flip the tensor
    rev = torch.flip(x, dims=[0])
    # find first position where it's not mask_value
    non_mask = (rev != mask_value).nonzero(as_tuple=True)[0]
    if len(non_mask) == 0:
        return x.numel()  # all values are mask_value
    return non_mask[0].item()


def sync_hook_devices(model):
    for module in model.modules():
        if hasattr(module, "_hf_hook"):
            try:
                device = next(module.parameters()).device
            except StopIteration:
                continue  # module has no params
            module._hf_hook.execution_device = device
    return model

def find_target_position(label_seq, pred_seq, yes_token_id, no_token_id):
        """
        Find the last occurrence of yes/no token in labels and calculate 
        the corresponding prediction position.
        
        Args:
            label_seq: torch.Tensor, shape [seq_len]
            pred_seq: torch.Tensor, shape [seq_len]
            yes_token_id: int
            no_token_id: int
            
        Returns:
            tuple: (pred_position, last_target_pos) or (None, -1) if not found
        """
        # Find all positions of yes/no tokens in labels (excluding IGNORE_INDEX positions)
        valid_mask = (label_seq != IGNORE_INDEX)
        yes_positions = ((label_seq == yes_token_id) & valid_mask).nonzero(as_tuple=True)[0]
        no_positions = ((label_seq == no_token_id) & valid_mask).nonzero(as_tuple=True)[0]
        
        # Find the last occurrence of yes or no token
        last_yes_pos = yes_positions[-1].item() if len(yes_positions) > 0 else -1
        last_no_pos = no_positions[-1].item() if len(no_positions) > 0 else -1
        last_target_pos = max(last_yes_pos, last_no_pos)
        
        if last_target_pos == -1:
            return None, -1
        
        # Calculate position from the end
        seq_len = label_seq.shape[0]
        seq_len_pred = pred_seq.shape[0]
        position_from_end = seq_len - 1 - last_target_pos
        
        # Get the predicted token at the same position from the end
        pred_position = seq_len_pred - 1 - position_from_end
        
        return pred_position, last_target_pos

def make_compute_metrics_fn(tokenizer: transformers.PreTrainedTokenizer):
    """Make compute metrics function."""


    def compute_metrics(eval_preds):
        # preds, labels = eval_preds
        # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # # Some simple metrics
        # total = len(decoded_labels)
        # exact_match = 0
        # for pred, label in zip(decoded_preds, decoded_labels):
        #     if pred.strip() == label.strip():
        #         exact_match += 1
        # exact_match_rate = exact_match / total if total > 0 else 0.0

        # return {
        #     "exact_match": exact_match_rate,
        # }
        # Convert predictions (logits) into predicted token IDs
        ## chatgpt
        # preds = np.argmax(eval_preds.predictions, axis=-1)

        # # Map token IDs back to strings (using your tokenizer)
        # pred_tokens = [tokenizer.decode([pred_id], skip_special_tokens=True).strip() for pred_id in preds]
        # label_tokens = [tokenizer.decode([label_id], skip_special_tokens=True).strip() for label_id in eval_preds.label_ids]

        # # Compare predictions with labels
        # correct = sum(p == l for p, l in zip(pred_tokens, label_tokens))
        # accuracy = correct / len(label_tokens)
        preds, labels = eval_preds

        # handle model outputs that come as (logits, ...)
        if isinstance(preds, tuple):
            preds = preds[0]
        # # logits -> token ids
        # if getattr(preds, "ndim", 0) == 3:
        #     pred_ids = preds.argmax(axis=-1)
        # else:
        #     pred_ids = preds

        # pred_ids = np.asarray(pred_ids)
        label_ids = np.asarray(labels)

        pred_tokens = []
        label_tokens = []
        for l_seq in label_ids:
            mask = (l_seq != IGNORE_INDEX)
            if not mask.any():
                continue
            # pick last non-ignored token (common for generation labels)
            # idx = int(np.where(mask)[0][-1])
            # pred_tokens.append(int(p_seq[idx]))
            label_tokens.extend(l_seq[mask].tolist())

        # decode token ids to strings
        # pred_strs = [tokenizer.decode([t], clean_up_tokenization_spaces=True).strip().lower() for t in pred_tokens]
        label_strs = [tokenizer.decode([t], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().lower() for t in label_tokens]
        label_out = []

        def normalize_yesno(s):
            s = s.strip().lower()
            if "yes" in s:
                label_out.append(1)
            if "no" in s:
                label_out.append(0)
            return

        for l in label_strs:
            normalize_yesno(l)
        # pred_norm = [normalize_yesno(s) for s in pred_strs]

        if len(label_out) == 0:
            return {"accuracy": 0.0}
        acc = float(sum(p == l for p, l in zip(preds, label_out)) / len(label_out))

        return {"accuracy": acc}
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        logits: [batch_size, seq_len, vocab_size] or [batch_size, 1, seq_len, vocab_size]
        labels: [batch_size, seq_len] or [batch_size, 1, seq_len]
        
        For each example in the batch:
        1. Find the last occurrence of yes_token_id or no_token_id in labels
        2. Calculate its position from the end of the sequence
        3. Extract the predicted token at that same position from the end
        """
        yes_token_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
        no_token_id = tokenizer("no", add_special_tokens=False).input_ids[0]
        uc_yes_token_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        uc_no_token_id = tokenizer("No", add_special_tokens=False).input_ids[0]
        # Handle tuple outputs
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Squeeze extra dimensions if present
        if logits.dim() == 4:  # [batch_size, 1, seq_len, vocab_size]
            logits = logits.squeeze(1)
        if labels.dim() == 3:  # [batch_size, 1, seq_len]
            labels = labels.squeeze(1)
        
        # Get predicted token IDs from logits
        pred_tokens = logits.argmax(dim=-1).squeeze(0)  # [batch_size, seq_len]
        
        preds = []
        for batch_idx in range(labels.shape[0]):
            label_seq = labels[batch_idx]  # [seq_len]
            pred_seq = pred_tokens[batch_idx]  # [seq_len]
            
            # Find target position using extracted function
            pred_position, last_target_pos = find_target_position(
                label_seq, pred_seq, yes_token_id, no_token_id
            )
            
            

            if last_target_pos == -1:
                # No yes/no token found in labels, default prediction
                preds.append(2)  # Neutral/unknown
                continue
            
            predicted_token = pred_seq[pred_position-1].item()
            # print(tokenizer.decode(predicted_token))
            # Convert to binary prediction (1 for yes, 0 for no, 2 for other)
            if predicted_token == yes_token_id or predicted_token == uc_yes_token_id:
                preds.append(1)
            elif predicted_token == no_token_id or predicted_token == uc_no_token_id:
                preds.append(0)
            else:
                preds.append(2)  # Neither yes nor no

        
        return torch.tensor(preds, device=logits.device)
    
    return preprocess_logits_for_metrics, compute_metrics