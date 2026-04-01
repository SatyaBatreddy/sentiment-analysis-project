from transformers import pipeline

# Load DistilBERT sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

print("✅ Model loaded successfully!\n")

while True:
    text = input("Enter text (or type 'exit'): ")
    
    if text.lower() == 'exit':
        print("👋 Exiting...")
        break
    
    result = sentiment_pipeline(text)[0]
    
    print("\n📊 Sentiment:", result['label'])
    print("📈 Confidence:", round(result['score']*100, 2), "%\n")