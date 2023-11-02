
TOKENIZER_CACHE_DIR = './build/tokenizer_cache_dir/'
MODEL_CACHE_DIR = './build/model_cache_dir/'

CACHE_DIR = './build/cache_dir/'
DATA_DIR = './build/data_dir/'

from transformers import Trainer, TrainingArguments, LineByLineTextDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import json

model_name = "microsoft/DialoGPT-small"
model = GPT2LMHeadModel.from_pretrained(model_name,cache_dir=MODEL_CACHE_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(model_name,cache_dir=TOKENIZER_CACHE_DIR)

# Tokenize the data
tokenizer.pad_token = tokenizer.eos_token

# Load data
with open('00_dialogpt_data_example.json', 'r') as file:
    data = json.load(file)

# Prepare data for training
with open("./build/00_dialogpt_data_example.txt", "w") as f:
    for item in data:
        f.write("input: " + item["input"] + " response: " + item["response"] + '\n')
        # f.write("input: " + item["input"] + " response: " + item["response"] + tokenizer.eos_token + '\n')

# Create Dataset and DataLoader
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    block_size=128,
    file_path='./build/00_dialogpt_data_example.txt'
)

print("Number of samples in dataset:", len(dataset))


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=data_collator)

training_args = TrainingArguments(
    output_dir="./build/results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10,
    save_total_limit=2,
    report_to=[],  # Disable all integrations including wandb
    logging_steps=10,
    logging_dir='./build/logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

trainer.save_model("./build/models/dialogpt_model")
tokenizer.save_pretrained("./build/models/dialogpt_model")
