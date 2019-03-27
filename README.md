This repo contains Implementration of the paper 
### "Zero shot Multilingual sentiment analysis using Hierarchical Attentive BERT"


***** Follow the instruction of the below repository to install pretrained BERT module
https://github.com/huggingface/pytorch-pretrained-BERT

### clone thye following repository
https://github.com/openai/generating-reviews-discovering-sentiment

and replace the encoder.py , sst_binary_demo.py and utils.py code with ours..

### code for training version 1 ( without M-LSTM semantic feature) run
   python main.py

### code for training version 2 (with review level semantic features) run
  python main_sentiment.py

### For Inference of english reviews run
  python main_sentiment_predictor.py

### Train the model for languation translation pairs for hindi sentiment analysis run
  python srv_en_to_hindi_nn.py (just uncomment line 201 to 225)
