# Training customization

TRL is designed with modularity in mind so that users are able to efficiently customize the training loop for their needs. Below are examples on how you can apply and test different techniques. Note: Although these examples use the [`DPOTrainer`], these customization methods apply to most (if not all) trainers in TRL.

## Use different optimizers and schedulers

By default, the `DPOTrainer` creates a `torch.optim.AdamW` optimizer. You can create and define a different optimizer and pass it to `DPOTrainer` via the `optimizers` argument:

```python
from torch import optim

# Create a custom optimizer
optimizer = optim.SGD(model.parameters(), lr=training_args.learning_rate)

# Pass it to the trainer (optimizer, scheduler)
trainer = DPOTrainer(
    ...,
    optimizers=(optimizer, None),
)
```

### Add a learning rate scheduler

You can also play with your training by adding learning rate schedulers:

```python
from torch import optim

# Create optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=training_args.learning_rate)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Pass both to the trainer
trainer = DPOTrainer(
    ...,
    optimizers=(optimizer, lr_scheduler),
)
```

## Memory efficient fine-tuning by sharing layers

Another tool you can use for more memory efficient fine-tuning is to share layers between the reference model and the model you want to train:

```python
from trl import create_reference_model

# Create a reference model with shared layers
ref_model = create_reference_model(model, num_shared_layers=6)

# Pass it to the trainer
trainer = DPOTrainer(
    ...,
    ref_model=ref_model,
)
```

## Pass 8-bit reference models

Since `trl` supports all keyword arguments when loading a model from `transformers` using `from_pretrained`, you can also leverage `load_in_8bit` from `transformers` for more memory efficient fine-tuning.

Read more about 8-bit model loading in `transformers` [Load in 8bit or 4bit](https://huggingface.co/docs/transformers/en/peft#load-in-8bit-or-4bit).

```python
from transformers import BitsAndBytesConfig

# Create quantization config for 8-bit loading
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load reference model in 8-bit
ref_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quantization_config=quantization_config
)

# Pass it to the trainer
trainer = DPOTrainer(
    ...,
    ref_model=ref_model,
)
```

## Add custom callbacks

You can customize the training loop by adding callbacks for logging, monitoring, or early stopping:

```python
from transformers import TrainerCallback

# Define a custom callback
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")

# Pass it to the trainer
trainer = DPOTrainer(
    ...,
    callbacks=[CustomLoggingCallback()],
)
```

## Add custom evaluation metrics

You can define custom evaluation metrics to track during training:

```python
# Define a custom metric function
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Add your metric computation here
    return {"custom_metric": 0.0}

# Enable evaluation and pass the metric function
training_args = DPOConfig(
    ...,
    eval_strategy="steps",
    eval_steps=100
)

trainer = DPOTrainer(
    ...,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

## Use mixed precision training

Mixed precision training can significantly speed up training and reduce memory usage. **Note: TRL uses `bf16=True` by default**, which is optimal for modern GPUs with Ampere architecture or newer (A100, RTX 30xx/40xx).

You can override the default mixed precision settings if needed:

```python
# Override to use float16 for older GPUs that don't support bfloat16
training_args = DPOConfig(..., fp16=True, bf16=False)

# Or disable mixed precision entirely for full float32 training
training_args = DPOConfig(..., fp16=False, bf16=False)
```

## Use a custom data collator

You can provide a custom data collator to handle special data preprocessing or padding strategies:

```python
from trl.trainer.dpo_trainer import DataCollatorForPreference

# Create a custom data collator with specific padding token
data_collator = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)

# Pass it to the trainer
trainer = DPOTrainer(
    ...,
    data_collator=data_collator,
)
```
