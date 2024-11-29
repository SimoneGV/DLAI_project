import string
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset
import numpy as np
import re
from embeddings_generator import generate_embeddings, text_from_embeddings
from quantization_dequantization import quantization,dequantization
from evaluation import evaluation

def remove_punctuation(text):
    """Remove all whitespace characters from the text."""
    new_text = []
    for char in text:
        if char not in string.punctuation :
            new_text.append(char)
    return ''.join(new_text)


def from_letter_to_number(text):
    #Transform part of sentence into numbers
    # Create the regex patterns
    patterns = {
        r'\Bone\b': '1',
        r'\BOne\b': '1',
        r'\bto\b': '2',
        r'\bTo\b': '2',
        r'\Btwo\b': '2',
        r'\BTwo\b': '2',
        r'\Bthree\b': '3',
        r'\BThree\b': '3',
        r'\bfor\b': '4',
        r'\bFor\b': '4',
        r'\Bfour\b': '4',
        r'\BFour\b': '4',
        r'\bate\b': '8',
        r'\bAte\b': '8',
        r'\Beight\b': '8',
        r'\BEight\b': '8',
        
    }
    # Apply the patterns
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    
    return text
def remove_double_consonants(text):
    """Remove duplicate consecutive consonants from the text."""
    result = []
    for i in range(len(text)):
        # Check if the current character is a consonant and if it is equal to the previous character
        if i > 0 and text[i].lower() == text[i-1].lower() and text[i].lower() not in 'aeiou':
            continue
        result.append(text[i])
    return ''.join(result)



# Texts to be transformed into embeddings, then quantized and transformed back into texts
texts = [
    "Jack Morris is a PhD student at Cornell Tech in New York City",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity",
    # General Knowledge
    "The Eiffel Tower is one of the most famous landmarks in Paris, France.",
    "Albert Einstein developed the theory of relativity, fundamentally altering the understanding of physics.",
    # Literary Excerpt
    "In the beginning, God created the heavens and the earth.",
    "To be, or not to be, that is the question.",
    # Dialogue
    "Can you please pass the salt?",
    "Sure, I’ll meet you at the coffee shop at 3 PM.",
    # Technical/Scientific
    "Quantum entanglement describes a physical phenomenon where particles become interconnected and instantaneously affect each other, regardless of distance.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
    # Questions
    "What is the capital of Japan?",
    "How many continents are there on Earth?",
    # Short Sentences
    "The sun rises in the east.",
    "Water is essential for life.",
    # Complex Sentence
    "Despite the challenges posed by the rugged terrain, the explorers pressed on, determined to reach the summit before the storm hit.",
    "The conference on artificial intelligence attracted experts from various fields, sparking debates on the ethical implications of autonomous systems."
]


original_texts_size = sum(len(text.encode('utf-8')) for text in texts)
print("Size of original texts (in bytes):", original_texts_size)

texts_no_punct = [remove_punctuation(text) for text in texts]
print(texts_no_punct)
no_whitespace_texts_size = sum(len(text.encode('utf-8')) for text in texts_no_punct)
print("Size of texts with whitespace removed (in bytes):", no_whitespace_texts_size)

texts_lett_num = [from_letter_to_number(text) for text in texts_no_punct]
print(texts_lett_num)
no_whitespace_texts_size = sum(len(text.encode('utf-8')) for text in texts_lett_num)
print("Size of texts with whitespace removed (in bytes):", no_whitespace_texts_size)

texts_no_whitespace = [remove_double_consonants(text) for text in texts_lett_num]
print(texts_no_whitespace)
no_whitespace_texts_size = sum(len(text.encode('utf-8')) for text in texts_no_whitespace)
print("Size of texts with whitespace removed (in bytes):", no_whitespace_texts_size)

# Generate embeddings
embeddings = generate_embeddings(texts_lett_num)

print("Original embeddings:")
print(embeddings)
print("Size in bytes:", embeddings.nbytes)

# Quantize embeddings to int8
quantized_embeddings= quantization(embeddings)
print("Quantized embeddings:")
print(quantized_embeddings)
print("Size in bytes:", quantized_embeddings.nbytes)
# Dequantize embeddings back to float32
dequantized_embeddings = dequantization(quantized_embeddings)
print("Dequantized embeddings:")
print(dequantized_embeddings)

# Transform embeddings back into text
dequantized_text = text_from_embeddings(dequantized_embeddings)
print("Text from dequantized embeddings:")
print(dequantized_text)

cos_sim, euclidean, bleu, avg_bleu, rouge, edit_dist, jaccard = evaluation(embeddings,dequantized_embeddings,texts_lett_num,dequantized_text)


print(f"Cosine Similarity between original and dequantized embeddings: {cos_sim:.4f}")

print(f"Euclidean Distance between original and dequantized embeddings: {euclidean:.4f}")

print(f"BLEU score between original and dequantized texts: {bleu}")

print(f"Average BLEU: {avg_bleu:.4f}")

print(f"ROUGE-1: {rouge['rouge1'].mid}")
print(f"ROUGE-2: {rouge['rouge2'].mid}")
print(f"ROUGE-L: {rouge['rougeL'].mid}")

print(f"Edit distance between texts and dequantized texts: {edit_dist}")

print(f"Jaccard distance between texts and dequantized texts: {jaccard}")
print(f"Average Jaccard distance between texts and dequantized texts: {np.mean(jaccard)}")

