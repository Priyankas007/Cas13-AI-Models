# Cas13 Artificial Intelligence (AI) Models
Repository of AI models developed and trained during Nov 2020 - June 2021 as an undergraduate researcher in the Stanley Qi Lab @ Stanford Bioengineering. All three models were coded in Python, utilizing the Keras, Pandas, and Tensorflow libraries, and Neptune.ai API for automated model and hyperparameter tuning. Due to a lack of Cas13 guide screenings being readily available, a synthetic dataset was created using RNAfold for model training, testing, and validation. 

### Minimum Free Energy (MFE) Prediction Recurrent Neural Network (RNN)
RNN that predicts the minimum free energy of Cas13 guide RNA (sgRNA) based on nucleotide sequence. LSTM layers were added to ensure the model was capable of learning order dependence for the 23-nucleotide long sgRNA used as inputs.  

### Guide Score Prediction Convolutional Neural Network (CNN)
A convolutional neural network (CNN) predicting guide scores for Cas13 sgRNA based on nucleotide sequence. I built upon the previous RNN framework by adding convolutional layers on top of the LSTM layers. After model training, this model was packaged into an API through the guidescore_model_development.py document.

### Seq2Seq Guide RNA Predictor
A preliminary Seq2Seq model that predicts the optimal Cas13 guide sgRNA based on input target region. 

