"""
Utility functions for financial sentiment analysis using the FinBERT model.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face Transformers for FinBERT
import torch                                                               # PyTorch for model inference
import gc                                                                  # Garbage collection for memory management


device = "cuda:0" if torch.cuda.is_available() else "cpu"   # Select device: use GPU if available, else fallback to CPU

# Load the FinBERT tokenizer and model onto the selected device
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# FinBERT label mapping (order is as per model's output)
labels = ["positive", "negative", "neutral"]


def estimate_sentiment(news):
    """
    Estimates the overall sentiment of a list of news headlines/articles.

    Args:
        news (List[str]): List of news headlines or article snippets.

    Returns:
        Tuple[float, str]: (confidence score, sentiment label)
            - confidence: Probability/confidence in predicted sentiment (float between 0 and 1)
            - sentiment: Predicted sentiment ("positive", "negative", or "neutral")
    """
    if not news:
        return 0, "neutral"
    try:
        # Tokenize input news list for the FinBERT model
        inputs = tokenizer(news, return_tensors="pt", 
                        padding=True, truncation=True,
                        max_length=512).to(device)
        
        # Disable gradient calculation for inference (memory/performance optimization)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate probabilities for each article/news item
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Compute confidence weights for each article
        weights = probs.max(dim=1).values
        
        # Aggregate the results across all articles, weighted by their confidence
        weighted_probs = (probs * weights.unsqueeze(1)).sum(dim=0)
        normalized_probs = weighted_probs / weights.sum()
        
        # Pick sentiment with the highest probability (confidence, label index)
        confidence, pred = torch.max(normalized_probs, dim=0)
        return confidence.item(), labels[pred.item()]

    except Exception as e:
        print(f"Sentiment estimation error: {e}")
        return 0.0, "neutral"
    
    finally:
        # Clear GPU memory and run garbage collection (avoids CUDA out of memory in long-running apps)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

# Test block for standalone usage
if __name__ == "__main__":
    # Example news headlines
    example_news = [
        "markets responded positively to the news!",
        "traders were pleasantly surprised!",
        "economic fears weighed down investor sentiment.",
        "no major movement in the market today."
    ]
    confidence, sentiment = estimate_sentiment(example_news)
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    print("CUDA available:", torch.cuda.is_available())