library(keras)

model <- load_model_hdf5("Data/keras_model.h5", compile = F)
#load_model_weights_hdf5(model, "Data/keras_model_weights.h5")                #not required...load_model_hdf5() loads the weights along with model
tokenizer <- load_text_tokenizer(filename = "Data/keras_text_tokenizer")
maxlen = 150
label = c('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')


comment_preds <- function(x){
  
  preds <- data.frame()
  if(x == ""){
    preds <- data.frame(toxic = 0, severe_toxic = 0, obscene = 0, insult = 0, threat = 0, identity_hate = 0)
    
  }else{
    X_test <- texts_to_sequences(tokenizer, list(x))
    X_test <- pad_sequences(X_test, maxlen = maxlen, padding = "post", value = 0, truncating = "post")
    preds <- model %>% predict(X_test)
    colnames(preds) <- label
    return(preds)
  }
}

