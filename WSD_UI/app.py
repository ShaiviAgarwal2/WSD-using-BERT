import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel # Use Auto classes
import pickle
import re
import os
import nltk
from nltk.corpus import wordnet as wn

# --- Essential: Copy the WSDBertModel Class Definition ---
# Make sure this matches the class used during training *exactly*
class WSDBertModel(torch.nn.Module):
    def __init__(self, n_senses, model_name='bert-base-uncased'): # Ensure default model name matches
        super(WSDBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_senses)

    def forward(self, input_ids, attention_mask, target_token_index):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        # Use unsqueeze and expand for gathering - more robust
        idx = target_token_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, self.bert.config.hidden_size)
        target_token_embeddings = torch.gather(last_hidden_state, 1, idx).squeeze(1)
        # Alternative: might work if target_token_index shape is correct [batch_size]
        # target_token_embeddings = last_hidden_state[torch.arange(batch_size), target_token_index]
        dropped_output = self.dropout(target_token_embeddings)
        logits = self.classifier(dropped_output)
        return logits
# --- End of Model Class Definition ---

# --- Configuration ---
MODEL_PATH = "wsd_bert_model.bin"
MAPPINGS_PATH = "sense_mappings.pkl"
MODEL_NAME = 'bert-base-uncased' # Or distilbert if you switched
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Artifacts using Streamlit Cache ---
@st.cache_resource
def load_artifacts():
    print("Loading artifacts...")
    # Load NLTK data
    try:
        _ = wn.synsets("bank")
    except LookupError:
        print("Downloading NLTK WordNet data...")
        nltk.download('wordnet')
    try:
         _ = nltk.word_tokenize("test sentence")
    except LookupError:
        print("Downloading NLTK Punkt data...")
        nltk.download('punkt')


    if not os.path.exists(MAPPINGS_PATH):
        st.error(f"Error: Mappings file not found at {MAPPINGS_PATH}")
        return None, None, None, None

    if not os.path.exists(MODEL_PATH):
         st.error(f"Error: Model file not found at {MODEL_PATH}")
         return None, None, None, None

    with open(MAPPINGS_PATH, 'rb') as f:
        sense_to_id, id_to_sense = pickle.load(f)

    num_senses = len(sense_to_id)
    if num_senses == 0:
        st.error("Error: Loaded mappings contain no senses.")
        return None, None, None, None

    # Use AutoTokenizer to automatically get the Fast version
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Load model structure
    model = WSDBertModel(n_senses=num_senses, model_name=MODEL_NAME)
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Artifacts loaded successfully.")
    return model, tokenizer, sense_to_id, id_to_sense

# --- Prediction Function (Revised for Robustness) ---
def predict_sense(sentence, target_word, model, tokenizer, sense_to_id, id_to_sense, device):
    if not model or not tokenizer or not sense_to_id or not id_to_sense:
         st.error("Prediction function called before artifacts were loaded.")
         return None, None

    model.eval() # Ensure model is in eval mode

    sentence_lower = sentence.lower()
    target_word_lower = target_word.lower().strip()

    # --- Find the *first* occurrence's character offset ---
    start_char = -1
    try:
        # Use regex to find the first occurrence and its start char, trying with word boundaries first
        pattern = r'\b' + re.escape(target_word_lower) + r'\b'
        match = re.search(pattern, sentence_lower)

        if not match:
            # Fallback: try without word boundaries (e.g., if target is part of hyphenated word)
            pattern_no_boundary = re.escape(target_word_lower)
            match = re.search(pattern_no_boundary, sentence_lower)
            if not match:
               st.warning(f"Could not find the target word '{target_word}' in the sentence.")
               print(f"Debug: Could not find '{target_word_lower}' in '{sentence_lower}'")
               return None, None
            else:
               st.info(f"Note: Found '{target_word}' as a potential substring match.")

        start_char = match.start()
        print(f"Found target '{target_word}' starting at char index {start_char}")

    except Exception as e:
        st.error(f"An error occurred while finding the target word: {e}")
        print(f"Debug: Error finding '{target_word_lower}' in '{sentence_lower}'. Error: {e}")
        return None, None
    # --- End Finding Character Offset ---

    # Tokenize using the Fast Tokenizer
    try:
        encoding = tokenizer.encode_plus(
            sentence, # Use original case sentence for offset mapping consistency
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True, # Needs Fast Tokenizer
            return_tensors='pt',
        )
    except Exception as e:
        st.error(f"Error during tokenization: {e}")
        return None, None

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'].squeeze(0).tolist() # Remove batch dim correctly

    # --- Map start_char to token index ---
    target_token_idx = -1
    try:
        # Calculate end character based on the length of the *input* target word
        end_char = start_char + len(target_word_lower)

        found = False
        # Find the *first* token index whose start character matches `start_char`
        for idx, (start, end) in enumerate(offset_mapping):
            if start == start_char and end > start:
                # Check if the end offset roughly matches too, helps avoid partial matches
                if end >= end_char: # Token must cover at least the start of the word
                    target_token_idx = idx
                    found = True
                    print(f"Mapped char {start_char} to token {idx} with offsets {start, end}")
                    break
        # Fallback: find first token *starting within* the character span
        if not found:
            print(f"Debug: Exact start match failed for char {start_char}. Trying fallback...")
            for idx, (start, end) in enumerate(offset_mapping):
                 # Check if token *starts* within the word's span
                 if start >= start_char and start < end_char and end > start:
                     target_token_idx = idx
                     found = True
                     print(f"Used fallback offset mapping: Found token {idx} starting within span [{start_char}, {end_char}]")
                     break

        if not found:
            # Give more specific warning if mapping failed
            st.warning(f"Could not map target word '{target_word}' (found at char {start_char}) to any specific token.")
            print(f"Debug: Mapping failed. Offsets: {offset_mapping}, TargetSpan: [{start_char}, {end_char}]")
            return None, None # Cannot proceed

    except Exception as e:
         st.error(f"An error occurred during token index mapping: {e}")
         print(f"Debug: Error during token index mapping. Error: {e}")
         return None, None
    # --- End mapping ---

    target_token_index_tensor = torch.tensor([target_token_idx], dtype=torch.long).to(device)

    # Perform inference
    try:
        with torch.no_grad():
            logits = model(input_ids, attention_mask, target_token_index_tensor)

        probabilities = torch.softmax(logits, dim=1)
        predicted_sense_id = torch.argmax(probabilities, dim=1).item()

        # Map ID back to sense key
        predicted_sense_key = id_to_sense.get(predicted_sense_id, "Unknown Sense ID")

    except Exception as e:
        st.error(f"An error occurred during model prediction: {e}")
        print(f"Debug: Error during model forward pass or softmax/argmax. Error: {e}")
        return None, None


    # Try to get WordNet Synset for definition
    predicted_synset = None
    predicted_definition = "Could not retrieve definition."
    try:
        lemma = wn.lemma_from_key(predicted_sense_key)
        predicted_synset = lemma.synset()
        predicted_definition = predicted_synset.definition()
    except Exception as e:
        print(f"Could not get synset/definition from key {predicted_sense_key}: {e}")
        # Keep the sense key even if definition fails
        predicted_synset = f"WordNet lookup failed for key: {predicted_sense_key}"


    return predicted_sense_key, predicted_definition


# --- Streamlit UI ---
st.set_page_config(layout="wide") # Use wider layout

st.title("🧠 Word Sense Disambiguation (WSD)")
st.markdown("Enter a sentence and the target word you want to disambiguate.")

# Load model and other artifacts
model, tokenizer, sense_to_id, id_to_sense = load_artifacts()

if model: # Only proceed if artifacts loaded successfully
    col1, col2 = st.columns(2)

    with col1:
        sentence = st.text_area("Sentence:", height=100, placeholder="E.g., I went to the bank to deposit money.")
        target_word = st.text_input("Target Word:", placeholder="E.g., bank")

    disambiguate_button = st.button("Disambiguate", type="primary")

    st.divider() # Visual separator

    # Placeholder for results
    result_placeholder = st.empty()

    if disambiguate_button:
        if sentence and target_word:
            # Basic check if target word is likely in sentence (using simple check)
            if target_word.lower().strip() not in sentence.lower():
                 result_placeholder.warning(f"The target word '{target_word}' might not be present in the sentence as typed. Please check.")
            else:
                 result_placeholder.markdown("#### Prediction:")
                 with st.spinner("Predicting sense..."):
                    try:
                        sense_key, definition = predict_sense(
                            sentence, target_word, model, tokenizer, sense_to_id, id_to_sense, DEVICE
                        )

                        if sense_key:
                            # Use the placeholder to show results
                            result_placeholder.success(f"**Predicted Sense Key:** `{sense_key}`")
                            # Add definition below the success message
                            st.info(f"**Predicted Definition:** {definition}")
                        else:
                            # Error message already shown by predict_sense via st.warning/st.error
                            # Clear the placeholder if prediction failed internally
                            result_placeholder.empty()
                    except Exception as e:
                        st.error(f"An unexpected error occurred during prediction: {e}")
                        # Log the full traceback here for debugging if needed
                        # import traceback
                        # print(traceback.format_exc())
        else:
            result_placeholder.warning("Please enter both a sentence and a target word.")
else:
     st.error("Model artifacts could not be loaded. Cannot start the application.")


# Add some info/footer
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a BERT-based model fine-tuned on SemCor "
    "to predict the meaning (WordNet sense) of a word in context."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Model Used: `" + MODEL_NAME + "`")
st.sidebar.markdown("Device Used: `" + str(DEVICE) + "`")
