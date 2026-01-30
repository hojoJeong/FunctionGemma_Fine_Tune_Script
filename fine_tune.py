import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig
from load_model import model_id, model_path

output_dir = model_path  # Where to save your fine-tuned checkpoints
tokenizer = AutoTokenizer.from_pretrained(model_id)

args = SFTConfig(
    output_dir=output_dir,                            # Directory to save adapters
    num_train_epochs=2,                               # Number of training epochs
    per_device_train_batch_size=4,                    # Batch size per device during training
    gradient_accumulation_steps=8,                    # Gradient accumulation during training
    logging_strategy="steps",                         # Log every steps
    eval_strategy="steps",                            # Evaluate loss metrics based on steps
    eval_steps=50,                                    # Evaluate loss metrics every 50 steps
    logging_steps=50,                                 # Log loss metrics every 50 steps
    save_strategy="epoch",                            # Save checkpoint every epoch
    learning_rate=1e-5,                               # Learning rate,
    lr_scheduler_type="cosine",                       # Cosine scheduler is often better for full FT
    max_length=max_token_count,                       # Max sequence length for model and packing of the dataset
    gradient_checkpointing=True,                      # Use gradient checkpointing to save memory
    packing=False,                                    # Groups multiple samples in the dataset into a single sequence
    optim="adamw_torch_fused",                        # Use fused adamw optimizer
    bf16=True,                                        # Use bf16 for mixed precision training
    completion_only_loss=True,                        # Train on completion only to improve quality
    report_to="none"                                  # No reporting.
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation='eager')

base_model.config.pad_token_id = tokenizer.pad_token_id

print("Training configured")