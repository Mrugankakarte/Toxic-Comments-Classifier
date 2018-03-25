---
layout: post
title:  "Toxic Comment Classifier"
date:   2018-03-24 14:49:31 +0530
categories: jekyll update
---

This project is based on Kaggle Competition: Toxic Comment Classification Challenge

The challenge was to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. Comments from Wikipedia’s talk page edits were used as dataset to train models.

This was my first NLP Competition on Kaggle. Since everything was new to me, I learned a lot of new concepts, terms and techniques during the competiton.

Link to competition: [Toxic Comments Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Exploratory Data Analysis

*Disclaimer: Following content contains words that may be considered as vulger, profane or offensive.*

- The dataset was made up of 8 columns, id, Comments and 6 categories; toxic, severe_toxic, obscene, threat, insult and identity_hate.
![Dataset](/assets/Data_ss.PNG)

- Dimensions of training set = 159571 x 8 and dimensions of Testing set = 153164 x 2

- Most frequent words in training set are shown in bar graph
![freq_words](/assets/Rplot.png)

- Following graph shows most common words in each category.
![multi-plot](/assets/Rplot01.png)

## Model

Initially simple logistic regression was used to predict the probabilty of class being 1. Six such models were trained (one for each class). This model was used as baseline model. The input data to logistic regression was cleaned i.e numbers, english stopwords, puntuactions extra white spaces, Non-ASCII characters were removed, words were stemmed and converted to lower case. 

In the era of Deep learning, I decided to train neural networks for this task. Special type of neural networks called Recurrent neural networks are used to solve problems which exhibit behaviour of time sequence. Keras, which is based on Tensorflow framework provides lot of options, from tokenizing sentences to training a RNN on GPU, to solve NLP related problem. Recurrent nets are used to find patterns in sequence of data such as text, time series data, speech recognition, etc. The advatage of using recurrent net over feed forward net for such tasks is that unlike feed forward nets, recurrent nets possess some sort of memory which takes into account historical events to predict the future.

Final models submitted to the competition were GRU model and TextCNN. Average of two models scored AUC of 0.9786 on test data(Private Leaderboard 2708/4551) 

Following diagram shows the architechture of GRU and TextCNN model.
Pre-trained Word embedding used was Glove-twitter-27B_200d, maxlength of sentence: 150, vocabulary size: 30000  
The following model architecture was developed with the help of kaggle kernals posted by Kaggle Masters and Keras documentation site.

- __GRU__   
GRU units: 256, Dense units: 128, dropout: 0.2, learning-rate: 0.65, batch-size: 128, epoch: 4
![gru](/assets/gru.PNG)

- __TextCNN__  
Filter-size: (1,2,3,5), Filters: 32, dropout: 0.2, learning-rate: 0.003, batch size=64, epoch=10, patience=3
![textcnn](/assets/textcnn.png)

These models with parameters were finalized after running many iterations with different values for each parameter. I tried 5-Fold cross validation but a single run took more than 5hrs. Due to computation limitations, I decided to skip cross-validation approach and use only hold out set to determine the model performance.

## Metric

AUC-ROC (Area under curve-receiver operating curve) was metric used for this problem.
ROC curve is a plot of True positive rate vs False positive rate. True positive rate defines number of correct positive results among all positive samples, similarly false positive rate defines number of incorrect positive results among all negative samples.
Diagonal represents 50% probability; it is no better than random chance. Points above diagonal represent good classification result that is better than random.

### Long-Short Term Memory Cells

Long-Short Term Memory(LSTM) are building units from layers of Recurrent neural networks. It is composed of _Cell State_, _Input Gate_, _Output Gate_ and _Forget gate_. Cell state carries information from one cell to another. Forget gate is made up of sigmoid layer. It outputs the value between 0-1 to determine how much information to store from previous cell state. Input gate determines what new information to store in cell state. It is made up of two layers:
- sigmoid layer: This layer determines which value will update from old cell state
- tanh layer: This layer creates candidate values to add to cell state

The new cell state is calculated as,   
__old cell state x forget gate + input gate__   
where input gate = sigmoid layer x tanh layer. Finally in output layer, sigmoid layer decides which part of cell state is going to output. Then cell state is passed through tanh layer to push the vales between (-1,1) and then multplied with output of sigmoid gate.

### Gated Recurrent Units(GRU)

Gated Recurrent Units are based on LSTM(Long-Short Term Memory) cells. The difference between LSTM and GRU are as follows:

- Input and Output gates are combined into update gate
- Cell state and Hidden state are merged together


## Shiny App

I made a application which shows how toxic the sentence is in real time. I couldn't host it since the models were trained on GPU. Following are few screenshots of application.

![demo1](/assets/demo1.PNG)
![demo2](/assets/demo2.PNG)

### Conclusion

Following are the concepts related to NLP that I learned during this competiton.
- weighted Tfidf
- TermDocumentMatrix/DocumentTermMatrix
- WordEmbeddings
- RNN: GRU and LSTM
- Implementation of GRU and LSTM with hyperparameter tuning and cross-validation
- Implementation of CNN for text Classification
- Metric: AUC-ROC
- Always trust your cv score rather than public leaderboard score

There was lot of confusion regarding overfitting on leaderboard due lots of _blends of blends_ models. Hence I decided to stick with my own models rather than blends of blends and thought this would help me when final results come out. But unfortunately the blends were not overfitted and I went down the leaderboard. It was nice to explore the field which was completely new to me. Moving on to next competetion _TalkingData AdTracking Fraud detection Challenge_... 
