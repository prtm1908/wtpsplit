import re
from wtpsplit import SaT

def preprocess_text(text):
    """
    Remove capital letters and punctuation marks from text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation marks (keeping spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def segment_sentences(paragraph):
    """
    Process paragraph and segment into sentences using SaT model.
    """
    # Preprocess the text
    processed_text = preprocess_text(paragraph)
    
    # Initialize SaT model (12l-sm variant)
    sat = SaT("sat-12l-sm")
    
    # Optionally run on GPU for better performance
    # sat.half().to("cuda")
    
    # Split the processed text into sentences
    sentences = sat.split(processed_text)
    
    return sentences

# Example usage
if __name__ == "__main__":
    # Example paragraph
    paragraph = """The young Princess Bolkónskaya had brought some work in a gold-embroidered velvet bag. Her pretty little upper lip, on which a delicate dark down was just perceptible, was too short for her teeth, but it lifted all the more sweetly, and was especially charming when she occasionally drew it down to meet the lower lip. As is always the case with a thoroughly attractive woman, her defect—the shortness of her upper lip and her half-open mouth—seemed to be her own special and peculiar form of beauty. Everyone brightened at the sight of this pretty young woman, so soon to become a mother, so full of life and health, and carrying her burden so lightly. Old men and dull dispirited young ones who looked at her, after being in her company and talking to her a little while, felt as if they too were becoming, like her, full of life and health. All who talked to her, and at each word saw her bright smile and the constant gleam of her white teeth, thought that they were in a specially amiable mood that day."""
    
    print("Original paragraph:")
    print(paragraph)
    print("\n" + "="*50 + "\n")
    
    print("Processed text (lowercase, no punctuation):")
    processed = preprocess_text(paragraph)
    print(processed)
    print("\n" + "="*50 + "\n")
    
    print("Segmented sentences:")
    sentences = segment_sentences(paragraph)
    
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")