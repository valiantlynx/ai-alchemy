import os
import json
import keras
import keras_nlp

os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_KEY"]

os.environ["KERAS_BACKEND"] = "torch"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

data = []
with open("databricks-dolly-15k.jsonl") as file:
    for line in file:
        features = json.loads(line)
        if features["context"]:
            continue
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        data.append(template.format(**features))

data = data[:1000]

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()

print("Before fine-tuning:\n\n")

prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

gemma_lm.preprocessor.sequence_length = 512
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(data, epochs=1, batch_size=1)

model_save_kwargs = {
    "save_format": "tf",  # Specifies to save in TensorFlow format; alternatives include 'h5' for HDF5
    "include_optimizer": True,  # Whether to save the optimizer's state as well
}

gemma_lm.save('finetuned_model.keras', save_format="tf", include_optimizer=True)

# Now, when pushing to the Hugging Face Hub
from huggingface_hub import push_to_hub_keras

push_to_hub_keras(
    gemma_lm,
    "valiantlynx/gemma-2b-en-finetuned-databricks-dolly-15k",
    tags=["gemma-2b-en", "finetuned", "databricks-dolly-15k", "gemma", "lora"],
    **model_save_kwargs  # This expands to fill in the save arguments for the model
)

print("After fine-tuning:\n")

prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))
