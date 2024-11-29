from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
import vec2text

# Use the pretrained encoder
encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")

# Use the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

# Use the pretrained corrector
corrector = vec2text.load_pretrained_corrector("gtr-base")
def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length").to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings



def generate_embeddings(texts):
    embeddings = get_gtr_embeddings(texts, encoder, tokenizer)
    return embeddings


def text_from_embeddings(dequantized_embeddings):
    dequantized_text = vec2text.invert_embeddings(
    embeddings=dequantized_embeddings.cuda(),
    corrector=corrector,
    num_steps=50,
    )
    return dequantized_text