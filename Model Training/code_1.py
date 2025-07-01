import nltk
from nltk.corpus import semcor, wordnet as wn
from transformers import BertTokenizerFast, BertModel
from transformers.optimization import get_linear_schedule_with_warmup # Corrected line
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from tqdm.auto import tqdm
import re
import os
import pickle # To save mappings

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 8 # Adjust based on performance and time
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Data Loading and Preprocessing ---

def preprocess_semcor():
    """
    Loads SemCor, extracts sentences with tagged words, target word info, and sense keys.
    Returns a list of dictionaries, each representing a training instance.
    """
    print("Loading and preprocessing SemCor...")
    instances = []
    # Using tagged_sents with 'sem' tag gives Trees; 'word' gives flat lists
    # We need the structure to easily identify lemma and word form
    raw_sentences = semcor.tagged_sents(tag='sem')

    for sent_tree in tqdm(raw_sentences, desc="Processing SemCor Sentences"):
        current_words = []
        current_lemmas = []
        current_tags = [] # Store PoS tags if needed later
        target_indices = [] # Store indices of words that are sense-tagged
        sense_keys = []   # Store the corresponding sense keys


                # sent_tree is the SemcorSentence object - iterate through its elements (likely chunks/Trees)
        tagged_words_in_sent = []
        for element in sent_tree:
            if isinstance(element, nltk.Tree):
                # If the element is a Tree (like a chunk), get its tagged words
                tagged_words_in_sent.extend(element.pos())
            elif isinstance(element, tuple) and len(element) == 2:
                # Handle cases where it might be a direct (word, tag) tuple (less common with tag='sem')
                 tagged_words_in_sent.append(element)
            # Else: Handle other potential structures if needed, or ignore elements that aren't Trees/tuples

        # Now 'tagged_words_in_sent' contains the flat list of (word, tag) tuples for the whole sentence

        current_words = []
        current_lemmas = []
        current_tags = [] # Store PoS tags if needed later
        target_indices = [] # Store indices of words that are sense-tagged
        sense_keys = []   # Store the corresponding sense keys

        original_word_index = -1
        # Iterate through the CORRECT list of (word, tag) pairs collected above
        for word, tag in tagged_words_in_sent:
            original_word_index += 1
            # Basic cleaning: lowercasing words for BERT uncased
            cleaned_word = word.lower()
            current_words.append(cleaned_word)

            if isinstance(tag, nltk.corpus.reader.wordnet.Lemma):
                # This word is sense-tagged!
                lemma_name = tag.name() # e.g., 'run.v.01.run'
                sense_key = tag.key() # e.g., 'run%2:38:00::' - better unique identifier
                # Get base lemma requires careful parsing if format varies
                try:
                    # Attempt to parse standard WordNet lemma format like 'run.v.01.run'
                    base_lemma = lemma_name.split('.')[0]
                except:
                    # Fallback if parsing fails (e.g., unexpected lemma name format)
                    base_lemma = word # Use the word itself as fallback

                current_lemmas.append(base_lemma)
                try:
                    current_tags.append(tag.synset().pos()) # Get PoS (e.g., 'v')
                except Exception as e:
                    # Handle cases where synset might be missing or problematic
                    print(f"Warning: Could not get POS for lemma {tag}. Error: {e}")
                    current_tags.append(None)

                target_indices.append(original_word_index)
                sense_keys.append(sense_key)
            else:
                # Word is not sense-tagged (might just have POS tag)
                current_lemmas.append(word) # Use the word itself as lemma placeholder
                # Optionally try to extract POS tag if 'tag' is a string
                if isinstance(tag, str):
                     current_tags.append(tag) # Or map Penn tags to WordNet tags if needed
                else:
                     current_tags.append(None)


        # Create instances for each tagged word in the sentence
        # This part of the code using target_indices should now work correctly
        for i in range(len(target_indices)):
            target_idx = target_indices[i]
            sense = sense_keys[i]
            lemma = current_lemmas[target_idx] # The lemma of the target word

            # Ensure target_idx is valid before accessing lists
            if target_idx < len(current_words) and target_idx < len(current_tags):
                instances.append({
                    'sentence': " ".join(current_words),
                    'words': list(current_words), # Store list of words for easier indexing
                    'target_word': current_words[target_idx],
                    'target_lemma': lemma,
                    'target_pos': current_tags[target_idx],
                    'target_index': target_idx, # Index in the *original* word list
                    'sense_key': sense
                })
            else:
                 print(f"Warning: Invalid target_idx {target_idx} encountered. Skipping instance.")
                 print(f"Sentence words: {current_words}")
                 print(f"Target indices found: {target_indices}")


    print(f"Found {len(instances)} sense-tagged word instances in SemCor.")
    return instances

# --- 2. Create Mappings and Dataset Class ---

def create_sense_mappings(instances):
    """ Creates mappings from sense keys to integer IDs and vice versa. """
    all_senses = set(inst['sense_key'] for inst in instances)
    sense_to_id = {sense: i for i, sense in enumerate(all_senses)}
    id_to_sense = {i: sense for sense, i in sense_to_id.items()}
    print(f"Created mappings for {len(all_senses)} unique senses.")
    return sense_to_id, id_to_sense

class WSDBertDataset(Dataset):
    def __init__(self, instances, tokenizer, sense_to_id, max_len):
        self.instances = instances
        self.tokenizer = tokenizer
        self.sense_to_id = sense_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        instance = self.instances[item]
        sentence = instance['sentence']
        target_word_index = instance['target_index']  # Index in the original word list
        sense_key = instance['sense_key']
        words = instance['words']

        # Tokenize the sentence
        # --- INSIDE __getitem__ ---
        # ... (get instance, sentence, target_word_index, words, sense_key)
        encoding = self.tokenizer.encode_plus(
            sentence,  # Use original case sentence for offset mapping consistency
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,  # Needs Fast Tokenizer
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        # Ensure correct offset mapping conversion
        offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()

        # --- New Robust Logic to find target_token_idx ---
        target_token_idx = -1  # Initialize to -1 (failure)
        target_word_lower = instance['target_word'].lower()  # Already lowercased from preprocess_semcor

        try:
            # Find start character position using the stored word index
            # NOTE: This still relies on the original word list and index.
            # Ensure preprocess_semcor accurately joins sentences and stores indices.
            start_char = -1
            current_char = 0
            sentence_rebuilt = " ".join(words)  # Rebuild sentence from word list used for indexing
            sentence_lower = sentence_rebuilt.lower()  # Use the same lowercasing

            for i, word in enumerate(words):
                word_len = len(word)
                # Find the word using the stored index
                if i == target_word_index:
                    # Double check the word matches what we expect
                    if word == target_word_lower:
                        start_char = current_char
                        break
                    else:
                        # This indicates a mismatch between the word list and the target word stored
                        print(f"DBG_WARN (getitem): Word list mismatch at index {target_word_index}. Expected '{target_word_lower}', got '{word}'. Sentence: '{sentence_rebuilt}'")
                        # Try finding the first occurrence instead as a fallback
                        start_char = sentence_lower.find(target_word_lower)
                        if start_char != -1:
                            print(f"DBG_WARN (getitem): Using fallback start_char {start_char} based on find().")
                            break  # Exit loop after finding target index

                # Move character pointer, accounting for space
                current_char += word_len + 1

            # If start_char wasn't found (e.g., word mismatch and find() failed)
            if start_char == -1:
                raise ValueError(f"Could not reliably determine start character for '{target_word_lower}' at index {target_word_index}.")

            # Map character span to token index
            end_char = start_char + len(target_word_lower)
            found_token = False

            # Try exact start match first
            for idx, (start, end) in enumerate(offset_mapping):
                if start == start_char and end >= end_char:  # Token starts exactly and covers word
                    target_token_idx = idx
                    found_token = True
                    break

            # Fallback: Token starts within the word span
            if not found_token:
                for idx, (start, end) in enumerate(offset_mapping):
                    if start >= start_char and start < end_char and end > start:  # Token starts inside the word
                        target_token_idx = idx
                        found_token = True
                        # Optionally print warning if fallback is used often
                        # print(f"DBG_INFO (getitem): Used fallback mapping for '{target_word_lower}' char {start_char} to token {idx}")
                        break

            # If still not found after fallbacks
            if not found_token:
                raise ValueError(f"Could not map target word '{target_word_lower}' (char span [{start_char}-{end_char}]) to any token.")

        except Exception as e:
            # Log the failure clearly!
            print(f"DBG_FAIL (getitem): Mapping failed for word '{instance['target_word']}' in sentence '{sentence}'. Error: {e}")
            # Default to CLS token ONLY as a last resort if you absolutely cannot skip the sample
            target_token_idx = 0  # Index 0 is the [CLS] token

        # --- End New Robust Logic ---

        # Get sense_id (handle potential missing keys gracefully)
        sense_id = self.sense_to_id.get(sense_key, -1)
        if sense_id == -1:
            print(f"DBG_WARN (getitem): Sense key '{sense_key}' not found in sense_to_id mapping! Skipping instance potentially.")
        # Decide how to handle: return None? Or a dummy value that gets filtered later?
        # Returning dummy values might be easier if DataLoader handles filtering None poorly.
        # For now, let it pass, but add filtering in the training loop if needed.

        # --- Return dictionary ---
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_token_index': torch.tensor(target_token_idx, dtype=torch.long),
            'sense_id': torch.tensor(sense_id, dtype=torch.long),
            # Optionally include original index/word for easier debugging in training loop
            'debug_target_word': instance['target_word'],
            'debug_sentence': sentence
        }


# --- 3. Model Definition ---

class WSDBertModel(torch.nn.Module):
    def __init__(self, n_senses, model_name=MODEL_NAME):
        super(WSDBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # Add dropout for regularization
        self.dropout = torch.nn.Dropout(p=0.2)
        # Classification layer: maps BERT hidden state to number of senses
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_senses)

    def forward(self, input_ids, attention_mask, target_token_index):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Get the hidden states of the last layer
        last_hidden_state = outputs.last_hidden_state

        # Extract the hidden state corresponding to the target word's first token
        # We need to gather the states based on the target_token_index for each item in the batch
        batch_size = input_ids.shape[0]
        # Create indices for gather: (batch_idx, target_token_idx, hidden_dim_idx)
        # We only need (batch_idx, target_token_idx) and then select all hidden dims
        idx = target_token_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.bert.config.hidden_size)
        # Gather requires index shape to match tensor dims except for the dim being indexed
        # Reshape target_token_index for gather: [batch_size, 1, hidden_size]
        target_token_embeddings = torch.gather(last_hidden_state, 1, idx).squeeze(1)

        # Alternative (simpler if target_token_index is guaranteed valid):
        # target_token_embeddings = last_hidden_state[torch.arange(batch_size), target_token_index]

        # Apply dropout and the classification layer
        dropped_output = self.dropout(target_token_embeddings)
        logits = self.classifier(dropped_output)

        return logits

# --- 4. Training and Evaluation Functions ---

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training Batch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_token_index = batch['target_token_index'].to(device)
        sense_ids = batch['sense_id'].to(device)

        # Skip batches where sense_id might be invalid (e.g., -1 if handling wasn't perfect)
        if torch.any(sense_ids < 0):
            print("Warning: Skipping batch with invalid sense IDs.")
            continue

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_token_index=target_token_index
        )

        loss = loss_fn(outputs, sense_ids)

        # Get predictions
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == sense_ids)
        losses.append(loss.item())

        # Backpropagation
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_token_index = batch['target_token_index'].to(device)
            sense_ids = batch['sense_id'].to(device)

            if torch.any(sense_ids < 0):
                 print("Warning: Skipping evaluation batch with invalid sense IDs.")
                 continue


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_token_index=target_token_index
            )

            loss = loss_fn(outputs, sense_ids)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == sense_ids)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)


# --- 5. Main Execution ---

if __name__ == "__main__":
    # --- Load and Prepare Data ---
    instances = preprocess_semcor()
    if not instances:
        print("No instances found. Exiting.")
        exit()

    # Create or load mappings
    mapping_path = "sense_mappings.pkl"
    if os.path.exists(mapping_path):
        print(f"Loading mappings from {mapping_path}")
        with open(mapping_path, 'rb') as f:
            sense_to_id, id_to_sense = pickle.load(f)
    else:
        sense_to_id, id_to_sense = create_sense_mappings(instances)
        with open(mapping_path, 'wb') as f:
            pickle.dump((sense_to_id, id_to_sense), f)
            print(f"Saved mappings to {mapping_path}")


    num_senses = len(sense_to_id)
    if num_senses == 0:
        print("Error: No senses found in the data.")
        exit()

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # Split data (adjust test_size as needed)
    train_instances, val_instances = train_test_split(instances, test_size=0.15, random_state=42)

    train_dataset = WSDBertDataset(train_instances, tokenizer, sense_to_id, MAX_LEN)
    val_dataset = WSDBertDataset(val_instances, tokenizer, sense_to_id, MAX_LEN)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- Initialize Model, Optimizer, Loss ---
    model = WSDBertModel(n_senses=num_senses).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1), # 10% warmup steps
        num_training_steps=total_steps
    )

    from collections import Counter
    import numpy as np
    import math # If using log weight

    print("Calculating sense weights...")
    # Count senses ONLY in the training data
    sense_counts = Counter(instance['sense_key'] for instance in train_instances)
    total_senses_in_train = sum(sense_counts.values())
    num_classes = len(sense_to_id) # Total unique senses from mapping
    class_weights = torch.ones(num_classes, dtype=torch.float) # Default weight is 1

    print(f"Num classes: {num_classes}, Total senses in train: {total_senses_in_train}")

    not_found_in_train = 0
    for sense_key, sense_id in sense_to_id.items():
        count = sense_counts.get(sense_key, 0) # Get count from training data, default 0
        if count > 0:
            # Inverse frequency weighting: N / (C * Nc) - balances classes
            weight = total_senses_in_train / (num_classes * count)
            # Alternative: Log weighting - less extreme than inverse frequency
            # weight = math.log(1.0 * total_senses_in_train / count)
            class_weights[sense_id] = weight
        else:
            # Handle senses present in validation/overall but NOT in training split
            # Assign a high weight, or average weight, or default weight 1?
            # Assigning a high weight might be risky if it never appears. Default 1 is safer.
            class_weights[sense_id] = 1.0 # Keep default weight 1 if not seen in training
            not_found_in_train += 1

    if not_found_in_train > 0:
        print(f"Warning: {not_found_in_train} senses from the mapping were not found in the training split.")

    # Ensure weights are on the correct device
    class_weights = class_weights.to(DEVICE)
    print("Sense weights calculated and moved to device.")

    # --- Modify Loss Function Initialization ---
    # loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE) # <--- Comment out or delete old line
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

    # --- Training Loop ---
    best_accuracy = 0
    model_save_path = "wsd_bert_model.bin"

    for epoch in range(EPOCHS):
        print(f'--- Epoch {epoch + 1}/{EPOCHS} ---')

        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, DEVICE, scheduler, len(train_dataset)
        )
        print(f'Train loss {train_loss:.4f} | Train accuracy {train_acc:.4f}')

        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, DEVICE, len(val_dataset)
        )
        print(f'Val loss {val_loss:.4f}   | Val accuracy {val_acc:.4f}')
        print('-' * 20)

        if val_acc > best_accuracy:
            print(f"Accuracy improved ({best_accuracy:.4f} -> {val_acc:.4f}). Saving model...")
            torch.save(model.state_dict(), model_save_path)
            best_accuracy = val_acc

    print("Training complete.")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to {model_save_path}")

    # --- (Optional) Load Best Model for Inference Example ---
    print("\n--- Inference Example ---")
    # Load the best model state
    model = WSDBertModel(n_senses=num_senses) # Re-initialize model structure
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("Loaded trained model weights.")
    else:
        print("Could not find saved model weights for inference example.")
        exit()


    # --- Prediction Function ---
    def predict_sense(sentence, target_word, model, tokenizer, sense_to_id, id_to_sense, device, max_len=MAX_LEN):
        model.eval() # Ensure model is in eval mode

        # Preprocess the input sentence similar to training data
        words = sentence.lower().split() # Simple splitting, may need refinement
        try:
            # Find the index of the first occurrence of the target word
            target_index = -1
            current_pos = 0
            target_word_lower = target_word.lower()
            for i, word in enumerate(words):
                 # Need to handle potential mismatches if sentence preprocessing differs from target_word
                 # A simple check:
                 if word.strip('.,!?;:"\'()') == target_word_lower:
                     target_index = i
                     break
                 current_pos += len(word) + 1 # Assuming space separation

            if target_index == -1:
                print(f"Warning: Target word '{target_word}' not found precisely in sentence: '{sentence}'. Trying approximate match.")
                # Try finding as substring
                target_index = -1
                current_pos = 0
                for i, word in enumerate(words):
                    if target_word_lower in word.strip('.,!?;:"\'()'):
                        target_index = i
                        print(f"Found approximate match '{word}' at index {i}")
                        break
                    current_pos += len(word) + 1

                if target_index == -1:
                     raise ValueError(f"Target word '{target_word}' not found in sentence '{sentence}'")


        except ValueError as e:
            print(f"Error finding target word: {e}")
            return None, None


        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        offset_mapping = encoding['offset_mapping'].squeeze().tolist() # Remove batch dim

        # Find target token index (same logic as in Dataset)
        target_token_idx = -1
        word_start_char = None
        current_char_idx = 0
        # Use character offsets for better accuracy
        sentence_lower = sentence.lower()
        word_start_char = sentence_lower.find(target_word_lower, current_char_idx)

        # Adjust if word appears multiple times - here we assume the *first* occurrence identified by index
        temp_current_char_idx = 0
        for i in range(target_index):
             temp_current_char_idx = sentence_lower.find(words[i], temp_current_char_idx) + len(words[i]) + 1 # Move past word + space


        word_start_char = sentence_lower.find(target_word_lower, temp_current_char_idx)


        if word_start_char == -1:
             print(f"Warning: Could not precisely locate start character for '{target_word}' in '{sentence}'.")
             # Fallback based on word index (less reliable with complex tokenization/punctuation)
             current_char_idx_approx = 0
             for i in range(target_index):
                 current_char_idx_approx += len(words[i]) + 1
             word_start_char = current_char_idx_approx


        # Find the token corresponding to this character start
        for idx, (start, end) in enumerate(offset_mapping):
            if start == word_start_char and end > start:
                target_token_idx = idx
                break
        if target_token_idx == -1:
            # Fallback if exact start match fails
            word_end_char = word_start_char + len(target_word)
            for idx, (start, end) in enumerate(offset_mapping):
                if start >= word_start_char and start < word_end_char and end > start:
                    target_token_idx = idx
                    break
            if target_token_idx == -1:
                print(f"FATAL: Could not map target word '{target_word}' to tokens during prediction. Sentence: '{sentence}'")
                return None, None # Cannot proceed


        target_token_index_tensor = torch.tensor([target_token_idx], dtype=torch.long).to(device) # Needs batch dim

        with torch.no_grad():
            logits = model(input_ids, attention_mask, target_token_index_tensor)

        probabilities = torch.softmax(logits, dim=1)
        predicted_sense_id = torch.argmax(probabilities, dim=1).item()

        predicted_sense_key = id_to_sense.get(predicted_sense_id, "Unknown Sense ID")

        # Try to get WordNet Synset for definition (optional)
        predicted_synset = None
        try:
            # Extract lemma.pos.sense_num from key (heuristic, might fail for some keys)
            match = re.match(r'(.+)%(\d):(\d+):(\d+)::', predicted_sense_key)
            if match:
                lemma_str = match.group(1)
                # Map WordNet POS tags (1: NOUN, 2: VERB, 3: ADJECTIVE, 4: ADVERB, 5: ADJECTIVE SATELLITE)
                pos_map = {'1': wn.NOUN, '2': wn.VERB, '3': wn.ADJ, '4': wn.ADV, '5': wn.ADJ_SAT}
                wn_pos = pos_map.get(match.group(2))
                if wn_pos:
                     # Get all synsets for the lemma and POS
                     synsets = wn.synsets(lemma_str, pos=wn_pos)
                     # Find the synset matching the key (requires matching sense number etc., potentially complex)
                     # A simpler approach: just return the first synset if available
                     if synsets:
                         # This is NOT guaranteed to be the correct synset matching the key
                         # A direct lookup using Lemma Key is better if available in NLTK easily
                         # predicted_synset = wn.lemma_from_key(predicted_sense_key).synset() # This is the correct way!
                         try:
                            predicted_synset = wn.lemma_from_key(predicted_sense_key).synset()
                         except Exception as e:
                            print(f"Could not get synset from key {predicted_sense_key}: {e}")
                            predicted_synset = f"Synset lookup failed for key: {predicted_sense_key}"

            else:
                 predicted_synset = f"Could not parse key: {predicted_sense_key}"


        except Exception as e:
            print(f"Error getting WordNet Synset: {e}")
            predicted_synset = "WordNet lookup error"


        return predicted_sense_key, predicted_synset


    # --- Example Usage ---
    test_sentence = "I went to the bank to deposit money."
    test_target = "bank"
    predicted_key, predicted_synset = predict_sense(
        test_sentence, test_target, model, tokenizer, sense_to_id, id_to_sense, DEVICE
    )

    if predicted_key:
        print(f"\nSentence: '{test_sentence}'")
        print(f"Target Word: '{test_target}'")
        print(f"Predicted Sense Key: {predicted_key}")
        if predicted_synset and hasattr(predicted_synset, 'definition'):
            print(f"Predicted Sense Definition: {predicted_synset.definition()}")
        else:
            print(f"Predicted Synset Info: {predicted_synset}") # Print whatever info we got

    test_sentence_2 = "The river bank was eroded."
    test_target_2 = "bank"
    predicted_key_2, predicted_synset_2 = predict_sense(
        test_sentence_2, test_target_2, model, tokenizer, sense_to_id, id_to_sense, DEVICE
    )

    if predicted_key_2:
        print(f"\nSentence: '{test_sentence_2}'")
        print(f"Target Word: '{test_target_2}'")
        print(f"Predicted Sense Key: {predicted_key_2}")
        if predicted_synset_2 and hasattr(predicted_synset_2, 'definition'):
            print(f"Predicted Sense Definition: {predicted_synset_2.definition()}")
        else:
            print(f"Predicted Synset Info: {predicted_synset_2}")