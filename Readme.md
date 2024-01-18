# Stance View
Dataset used: [Fake News Challenge](https://www.kaggle.com/datasets/abhinavkrjha/fake-news-challenge/code)  

[Largent, W. (2017, June). Talos Targets Disinformation with Fake News Challenge Victory.] Talos. https://blog.talosintelligence.com/2017/06/talos-fake-news-challenge.html
# Emotion View
Dataset used: [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)  

[CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404) (Saravia et al., EMNLP 2018)
# Stance Analysis and Multihead
## Dataset1 
Dataset used: [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  

Ahmed H, Traore I, Saad S. “Detecting opinion spams and fake news using text classification”, Journal of Security and Privacy, Volume 1, Issue 1, Wiley, January/February 2018.

Ahmed H, Traore I, Saad S. (2017) “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138).

## Dataset2
Dataset used: [FC-dataset] https://www.kaggle.com/datasets/mdepak/fakenewsnet

Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. ACM SIGKDD explorations newsletter, 19(1), 22-36.

Shu, K., Wang, S., & Liu, H. (2017). Exploiting tri-relationship for fake news detection. arXiv preprint arXiv:1712.07709, 8.

Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2020). Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Big data, 8(3), 171-188.


**Models:** 
- [PyTorch-BiLSTM-with-MHSA](multi-head): A Bidirectional LSTM classifer implemented in PyTorch with a custom layer for computing the multi-headed self attention
- [PyTorch-TextCNN](emotion_view)
- [PyTorch-LSTM](stance_view)
- [PyTorch-MLP](style_view.py)


