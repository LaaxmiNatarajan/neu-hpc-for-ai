
import inspect
import torch
import transformers

from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    print("=" * 65)
    print("QWEN3 VECTOR BATCHING API INSPECTION")
    print("=" * 65)

    print("Transformers:", transformers.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    model.eval()

    print("\nMODEL CLASS")
    print(type(model))

    print("\nMODEL FORWARD")
    print(inspect.signature(model.forward))

    print("\nMODEL CLASS SOURCE")
    print(inspect.getsource(type(model).forward))

    inner_model = model.model

    print("\nINNER MODEL CLASS")
    print(type(inner_model))

    print("\nINNER MODEL FORWARD")
    print(inspect.signature(inner_model.forward))

    print("\nINNER MODEL SOURCE")
    print(inspect.getsource(type(inner_model).forward))

    print("\nATTENTION MODULE CLASS")
    attention = inner_model.layers[0].self_attn
    print(type(attention))

    print("\nATTENTION FORWARD")
    print(inspect.signature(attention.forward))

    print("\nATTENTION SOURCE")
    print(inspect.getsource(type(attention).forward))

    print("\nATTENTION ATTRIBUTES")
    for name in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "num_heads",
        "num_key_value_heads",
        "head_dim",
        "scaling",
    ]:
        print(name, getattr(attention, name, None))

    print("\nCONFIG")
    for name in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    ]:
        print(name, getattr(model.config, name, None))

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()
