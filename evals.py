import re
from typing import List, Tuple
from main import segment_sentences, preprocess_text

def create_ground_truth_sentences(paragraph: str) -> List[str]:
    """
    Create ground truth sentence segmentation based on periods/exclamation marks 
    followed by capital letters, ignoring whitespace.
    Also handles sentences ending with punctuation at the end of paragraph.
    Handles quotes after punctuation marks.
    """
    # Find all sentence boundaries: periods or exclamation marks followed by whitespace and capital letter
    sentence_boundaries = []
    
    # Pattern 1: [.!] followed by whitespace and capital letter
    for match in re.finditer(r'[.!]\s+(?=[A-Z])', paragraph):
        sentence_boundaries.append(match.end() - 1)  # Position of the punctuation mark
    
    # Pattern 2: [.!] followed by quotes and then whitespace and capital letter
    # This handles cases like: "Hello." "World" -> sentence ends after the quote
    for match in re.finditer(r'[.!][\'"]\s+(?=[A-Z])', paragraph):
        sentence_boundaries.append(match.end() - 1)  # Position after the quote
    
    # Also find periods or exclamation marks at the very end of the paragraph
    if paragraph.rstrip().endswith(('.', '!')):
        # Find the last period or exclamation mark
        last_period = paragraph.rfind('.')
        last_exclamation = paragraph.rfind('!')
        last_punct = max(last_period, last_exclamation)
        if last_punct != -1 and last_punct not in sentence_boundaries:
            sentence_boundaries.append(last_punct)
    
    # Sort boundaries to ensure correct order
    sentence_boundaries.sort()
    
    # Extract sentences based on boundaries
    sentences = []
    start = 0
    
    for boundary in sentence_boundaries:
        sentence = paragraph[start:boundary + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = boundary + 1
    
    # If there are no boundaries found, treat the entire paragraph as one sentence
    if not sentence_boundaries:
        sentences = [paragraph.strip()]
    
    return sentences

def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison by converting to lowercase and removing extra whitespace.
    """
    # Convert to lowercase and normalize whitespace
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    # Remove punctuation for comparison since SaT output doesn't have punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return normalized

def calculate_metrics(ground_truth: List[str], predictions: List[str]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for sentence segmentation.
    
    Args:
        ground_truth: List of ground truth sentences
        predictions: List of predicted sentences
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Normalize both lists for comparison
    gt_normalized = [normalize_text_for_comparison(s) for s in ground_truth]
    pred_normalized = [normalize_text_for_comparison(s) for s in predictions]
    
    # Count correct predictions (exact matches)
    correct = 0
    for pred in pred_normalized:
        if pred in gt_normalized:
            correct += 1
    
    # Calculate precision and recall
    precision = correct / len(pred_normalized) if pred_normalized else 0.0
    recall = correct / len(gt_normalized) if gt_normalized else 0.0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def calculate_cumulative_metrics(all_ground_truth: List[List[str]], all_predictions: List[List[str]]) -> Tuple[float, float, float]:
    """
    Calculate cumulative precision, recall, and F1 score across all paragraphs.
    
    Args:
        all_ground_truth: List of ground truth sentence lists for each paragraph
        all_predictions: List of predicted sentence lists for each paragraph
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    total_correct = 0
    total_predictions = 0
    total_ground_truth = 0
    
    for gt_sentences, pred_sentences in zip(all_ground_truth, all_predictions):
        # Normalize for comparison
        gt_normalized = [normalize_text_for_comparison(s) for s in gt_sentences]
        pred_normalized = [normalize_text_for_comparison(s) for s in pred_sentences]
        
        # Count correct predictions
        for pred in pred_normalized:
            if pred in gt_normalized:
                total_correct += 1
        
        total_predictions += len(pred_normalized)
        total_ground_truth += len(gt_normalized)
    
    # Calculate overall metrics
    precision = total_correct / total_predictions if total_predictions > 0 else 0.0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def evaluate_paragraph(paragraph: str) -> Tuple[List[str], List[str], float, float, float]:
    """
    Evaluate a single paragraph by comparing ground truth with SaT predictions.
    
    Returns:
        Tuple of (ground_truth_sentences, predicted_sentences, precision, recall, f1_score)
    """
    # Create ground truth
    ground_truth = create_ground_truth_sentences(paragraph)
    
    # Get predictions from SaT model
    try:
        predicted = segment_sentences(paragraph)
    except Exception as e:
        print(f"Error in SaT prediction: {e}")
        predicted = []
    
    # Calculate metrics
    precision, recall, f1_score = calculate_metrics(ground_truth, predicted)
    
    return ground_truth, predicted, precision, recall, f1_score

def load_test_data(file_path: str) -> List[str]:
    """
    Load paragraphs from test file, separated by double newlines.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines to get paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs

def main():
    """
    Main evaluation function.
    """
    print("Loading test data...")
    paragraphs = load_test_data('test_set.txt')
    print(f"Loaded {len(paragraphs)} paragraphs for evaluation.\n")
    
    all_ground_truth = []
    all_predictions = []
    
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for i, paragraph in enumerate(paragraphs, 1):
        print(f"\nParagraph {i}:")
        print("-" * 40)
        
        # Evaluate this paragraph
        gt_sentences, pred_sentences, precision, recall, f1_score = evaluate_paragraph(paragraph)
        
        # Store for cumulative metrics
        all_ground_truth.append(gt_sentences)
        all_predictions.append(pred_sentences)
        
        print(f"Ground Truth Sentences ({len(gt_sentences)}):")
        for j, sentence in enumerate(gt_sentences, 1):
            print(f"  {j}. {sentence}")
        
        print(f"\nPredicted Sentences ({len(pred_sentences)}):")
        for j, sentence in enumerate(pred_sentences, 1):
            print(f"  {j}. {sentence}")
        
        print(f"\nMetrics for Paragraph {i}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
    
    # Calculate cumulative metrics across all paragraphs
    cumulative_precision, cumulative_recall, cumulative_f1 = calculate_cumulative_metrics(all_ground_truth, all_predictions)
    
    print("\n" + "=" * 80)
    print("CUMULATIVE RESULTS")
    print("=" * 80)
    print(f"Cumulative Precision: {cumulative_precision:.4f}")
    print(f"Cumulative Recall: {cumulative_recall:.4f}")
    print(f"Cumulative F1 Score: {cumulative_f1:.4f}")
    print(f"Total Paragraphs Evaluated: {len(paragraphs)}")
    
    # Also show total sentence counts
    total_gt_sentences = sum(len(gt) for gt in all_ground_truth)
    total_pred_sentences = sum(len(pred) for pred in all_predictions)
    print(f"Total Ground Truth Sentences: {total_gt_sentences}")
    print(f"Total Predicted Sentences: {total_pred_sentences}")

def test_ground_truth():
    """
    Test function to verify ground truth sentence segmentation.
    """
    test_paragraph = """The young Princess Bolkónskaya had brought some work in a gold-embroidered velvet bag. Her pretty little upper lip, on which a delicate dark down was just perceptible, was too short for her teeth, but it lifted all the more sweetly, and was especially charming when she occasionally drew it down to meet the lower lip. As is always the case with a thoroughly attractive woman, her defect—the shortness of her upper lip and her half-open mouth—seemed to be her own special and peculiar form of beauty. Everyone brightened at the sight of this pretty young woman, so soon to become a mother, so full of life and health, and carrying her burden so lightly. Old men and dull dispirited young ones who looked at her, after being in her company and talking to her a little while, felt as if they too were becoming, like her, full of life and health. All who talked to her, and at each word saw her bright smile and the constant gleam of her white teeth, thought that they were in a specially amiable mood that day."""
    
    print("Testing ground truth sentence segmentation:")
    print("=" * 60)
    print("Original paragraph:")
    print(test_paragraph)
    print("\n" + "=" * 60)
    
    sentences = create_ground_truth_sentences(test_paragraph)
    print(f"Ground truth sentences ({len(sentences)}):")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")

if __name__ == "__main__":
    # Uncomment the line below to test ground truth creation
    # test_ground_truth()
    main()
