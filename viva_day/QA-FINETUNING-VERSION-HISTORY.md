

| Parameter | v2 | v6 |  |
|---|---|---|---|
| LoRA rank | 16 | 32 | Doubled the model's dedicated capacity for learning the QA task. |
| Epochs | fixed 3 | 3 (tested against 5) | Directly tested 5 vs. 3 epochs; 5 overfit — validation loss got worse partway through while training loss kept dropping — so 3 was kept. |
| Learning rate | 1e-4 | 8e-5 | Trained more slowly and carefully, reducing the risk of unstable updates. |
| Effective batch size | 16 (4×4) | 32 (2×16) | Averaged gradients over twice as many examples per update, giving steadier, less noisy training. |



```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                                    # <- LoRA rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    modules_to_save=["lm_head"]
)

training_args = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    per_device_train_batch_size=4,           # <- effective batch = 4 x 4 = 16
    gradient_accumulation_steps=4,           # <-
    learning_rate=1e-4,                      # <- learning rate
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=3,                      # <- epochs
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
    report_to="none",
)
```

**v6 — `llama-scripts/qa-finetuning_v6.ipynb`, cell 6, id `8c140aa8`** (starts `from peft
import LoraConfig, TaskType`):

```python
qa_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,                                    # <- LoRA rank
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

training_epochs = 2 if len(train_dataset) >= 50_000 else 3   # <- epochs (evaluates to 3)
print(f"Training epochs selected for dataset size: {training_epochs}")
...

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    max_length=MAX_LENGTH,
    completion_only_loss=True,
    packing=False,
    eval_packing=False,
    per_device_train_batch_size=2,           # <- effective batch = 2 x 16 = 32
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,          # <-
    learning_rate=8e-5,                      # <- learning rate
    num_train_epochs=training_epochs,        # <- epochs
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    ...
)

trainer = SFTTrainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],   # <- early stopping
)
```
