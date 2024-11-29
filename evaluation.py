import torch.nn.functional as F
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import nltk
from rouge_score import rouge_scorer, scoring


def cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(embeddings1, embeddings2).mean().item()


# Function to compute ROUGE scores
def compute_rouge_scores(reference_texts, generated_texts):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for ref, gen in zip(reference_texts, generated_texts):
        scores = scorer.score(ref, gen)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return result


def compute_euclidean_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    return torch.norm(emb1 - emb2, dim=-1).mean()



def compute_edit_distance(reference_texts, generated_texts):
    distances = []
    for ref, gen in zip(reference_texts, generated_texts):
        distance = nltk.edit_distance(ref, gen)
        distances.append(distance)
    return distances



def compute_jaccard_distance(reference_texts, generated_texts):
    distances = []
    for ref, gen in zip(reference_texts, generated_texts):
        # Convert strings to sets of words
        ref_set = set(ref.split())
        gen_set = set(gen.split())
        # Compute Jaccard distance
        intersection = len(ref_set.intersection(gen_set))
        union = len(ref_set.union(gen_set))
        distance = intersection/union
        distances.append(distance)
    return distances



def evaluation(embeddings, dequantized_embeddings,texts, deq_texts):
    cos_sim = cosine_similarity(embeddings,  dequantized_embeddings.clone().detach().to("cuda"))
    euclidian = compute_euclidean_distance(embeddings.cpu(),dequantized_embeddings.cpu())
    smooth_fn = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([orig.split()], gen.split(), smoothing_function=smooth_fn) for orig, gen in zip(texts, deq_texts)]
    avg_bleu = np.mean(bleu_scores)
    rouge_scores = compute_rouge_scores(texts,deq_texts)
    edit_distance = compute_edit_distance(texts,deq_texts)
    jaccard_distance = compute_jaccard_distance(texts, deq_texts)

    return cos_sim, euclidian,  bleu_scores, avg_bleu,rouge_scores,edit_distance, jaccard_distance