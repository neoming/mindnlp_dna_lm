import mindspore
import mindnlp
import numpy as np
import tqdm
from mindnlp.transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")
lm = AutoModelForMaskedLM.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")
lm.eval()
from mindnlp.dataset import load_dataset
raw_data = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", "H3")
raw_data

train_dataset = raw_data['train']
test_dataset = raw_data['test']

def show_dataset_info(dataset):
    print("dataset column: {}".format(dataset.get_col_names()))
    print("dataset size: {}".format(dataset.get_dataset_size()))
    print("dataset batch size: {}\n".format(dataset.get_batch_size()))

print("train dataset info:")
show_dataset_info(train_dataset)
print("test dataset info:")
show_dataset_info(test_dataset)

for data in train_dataset:
    print(data)
    break

train_dataset, valid_dataset = train_dataset.split([0.85, 0.15])

print("train dataset info:")
show_dataset_info(train_dataset)
print("valid dataset info:")
show_dataset_info(valid_dataset)

def get_kmers(seq, k=6, stride=1):
    return [seq[i:i + k] for i in range(0, len(seq), stride) if i + k <= len(seq)]


BATCH_SIZE = 16
DATASET_LIMIT = 200 # NOTE: This dataset limit is set to 200, so that the training runs faster. It can be set to None to use the
                    # entire dataset

def process_sequence(data):
    sequence = data.tolist()
    sequence = "candida_glabrata " + " ".join(get_kmers(sequence))
    sequence = tokenizer(sequence)["input_ids"]
    return sequence

def nucleotide_dataset_process(dataset):
    # remove name column
    dataset = dataset.project(columns=['sequence', 'label'])

    # process sequence
    dataset = dataset.map(process_sequence, input_columns=['sequence'], output_columns=['sequence'])

    # change dataset size
    dataset = dataset.take(DATASET_LIMIT)

    # batch with padding
    dataset = dataset.padded_batch(batch_size=BATCH_SIZE,
                                   drop_remainder=True,
                                   pad_info={'sequence':(None, -100)})
    return dataset


train_dataset = train_dataset.apply(nucleotide_dataset_process)
valid_dataset = valid_dataset.apply(nucleotide_dataset_process)
test_dataset = test_dataset.apply(nucleotide_dataset_process)

print("train dataset info:")
show_dataset_info(train_dataset)

print("valid dataset info:")
show_dataset_info(valid_dataset)

print("test dataset info:")
show_dataset_info(test_dataset)

for data in train_dataset:
    print(data)
    break

import mindspore
from mindnlp.core import nn

class DNA_LM(nn.Module):
    def __init__(self, model, num_labels):
        super(DNA_LM, self).__init__()
        self.model = model.bert
        self.in_features = model.config.hidden_size
        self.out_features = num_labels
        self.classifier = nn.Linear(self.in_features, self.out_features)

    def forward(self, sequence, label=None):
        outputs = self.model(input_ids=sequence, attention_mask=None, output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1]
        # Use the [CLS] token for classification
        cls_output = sequence_output[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.out_features), label.view(-1))

        return (loss, logits) if loss is not None else logits

# Number of classes for your classification task
num_labels = 2
classification_model = DNA_LM(lm, num_labels)
classification_model

from mindnlp.engine import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_steps=1,
    logging_steps=1,
)

# Initialize Trainer
trainer = Trainer(
    model=classification_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()