import torch
import tiktoken
import argparse
import torch.nn.functional as F
from model import GPT, GPTConfig
from transformers import GPT2LMHeadModel

def test_gpt_model(model_type):
    """
    This function represents a Functional Test that loads weights from a pre-trained 
    GPT-2 model (from Hugging Face) and compares them with the weights of a custom GPT 
    model built from scratch. It checks whether the weights are successfully loaded 
    and the model behaves similarly to the GPT-2 model when performing text generation.

    Parameters:
    model_type (str): The type of GPT-2 model to compare. Options include 'gpt2', 'gpt2-medium', 
                      'gpt2-large', and 'gpt2-xl'.
    """

    print(f'GPT Model Test: Loading weights from pretrained {model_type}')

    # Define model configurations for different types of GPT-2 models
    config_args = {
        'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124 Million Params
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 358 Million Params
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774 Million Params
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558 Million Params
    }

    # Assert valid model type
    if model_type not in config_args:
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(config_args.keys())}.")

    config_args['vocab_size'] = 50257
    config_args['block_size'] = 1024
    config = GPTConfig(**config_args[model_type])

    # Create the GPT custom model 
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]  # Discard the mask keys


    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_hf_keys = [k for k in sd_hf.keys() if not (k.endswith('attn.bias') and k.endswith('.attn.masked_bias'))]

    # Check if the number of keys match
    assert len(sd_hf_keys) == len(sd_keys), f"mismatched keys: {len(sd_hf_keys)} != {len(sd_keys)}"

    # Define the layers that need transposing
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    # Compare the shapes and assign the weights
    for k, k_hf in zip(sd_keys, sd_hf_keys):
        if any(k_hf.endswith(w) for w in transposed):
            assert sd_hf[k_hf].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k_hf].shape[::-1]} != {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k_hf].t())  
        else:
            assert sd_hf[k_hf].shape == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k_hf].shape} != {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k_hf]) 

    # If we made it here then the test passed
    print(f"Successfully loaded weights for {model_type}. Test passed!")

    # --------------------------------------------------------------------------------------------------------------------
    num_return_sequences = 5
    max_length = 30
    seed = 42 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Start Evaluation and send Models to GPU
    model.eval()
    model.to(device)
    model_hf.to(device)
    # Encode the input string into tokens
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8)
    x = tokens.to(device)
    
    # Start The Generation Process
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    while x.size(1) < max_length:
        logits = model(x)[0] # (B,T,vocab_size)
        logits_hf = model_hf(x)[0]
        logits = logits[:, -1, :] # (B, vocab_size)
        logits_hf = logits_hf[:, -1, :] # (B, vocab_size)
        assert torch.allclose(logits, logits_hf, atol=1e-5), "Logits are not equal within tolerance!" 
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs,50, dim=-1) # (B,50)
        ix = torch.multinomial(topk_probs,1) # (B,1)
        xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        x = torch.cat((x,xcol), dim=-1) # (B, max_length)
    # print the generated text 
    for i in range(num_return_sequences):
        tokens = x[i,:].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
    print(f"Successfully Generated Sentences for {model_type} Matching the hugging face sentences. Test passed!")



if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description="Load GPT-2 model and verify weights.")
    parser.add_argument("model_type", type=str, help="Type of GPT-2 model: gpt2, gpt2-medium, gpt2-large, or gpt2-xl.")
    args = parser.parse_args()

    # Run the weight loading and comparison
    try:
        test_gpt_model(args.model_type)
    except Exception as e:
        print(f"Error during weight loading: {e}")
        exit(1)  

