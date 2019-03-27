This repo contains Implementration of the paper 
### "Zero shot Multilingual sentiment analysis using Hierarchical Attentive BERT"

### code for training version 1 ( without M-LSTM semantic feature) run
   python main.py

### code for training version 2 (with review level semantic features) run
  python main_sentiment.py

### For Inference of english reviews run
  python main_sentiment_predictor.py

### Train the model for languation translation pairs for hindi sentiment analysis run
  python srv_en_to_hindi_nn.py (just uncomment line 201 to 225)
