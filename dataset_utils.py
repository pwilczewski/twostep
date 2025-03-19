from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

def tokenize_function(examples, tokenizer):
    max_length = 512
    
    # removed padding
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    tokenized['labels'] = tokenized['input_ids'].clone()
    tokenized['input_ids'] = tokenized['input_ids'].squeeze()
    tokenized['attention_mask'] = tokenized['attention_mask'].squeeze()
    tokenized['labels'] = tokenized['labels'].squeeze()
    
    return tokenized

def load_tokenized_dataset(model_name, HF_TOKEN):

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("weaviate/wiki-sample", split="train", streaming=True)

    tokenized_dataset = dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        batch_size=8
    )

    return tokenized_dataset