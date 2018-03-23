This project is based on Kaggle Competition: Toxic Comment Classification Challenge

The challenge is to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. Comments from Wikipedia’s talk page edits was used as dataset to train models.

Link to competiton: [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

# Model

Final model selected for submission consists of GRU model and TextCNN. Average of two models scored AUC of 0.9786 on test data.

## Long-Short Term Memory Cells

Long-Short Term Memory(LSTM) are building units fro layers of Recurrent neural networks. It is composed of _Cell State_, _Input Gate_, _Output Gate_ and _Forget gate_. Cell state carries information from one cell to another. Forget gate is made up of sigmoid layer. It outputs the value between 0-1 to determine how much information to store from cell state.

## Gated Recurrent Units(GRU)

Gated Recurrent Units are based on LSTM(Long-Short Term Memory) cells. The difference between LSTM and GRU are as follows:
- Input and Output gates are combined into update gate
- Cell state and Hidden state are merged together

## Text

# Metric
AUC-ROC was metric used for this problem.
