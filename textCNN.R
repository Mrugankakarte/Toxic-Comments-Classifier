library(keras)

use_session_with_seed(seed = 123, disable_gpu = T, disable_parallel_cpu = F)
## EXPERIMENTAL ######
# auc_roc <- R6::R6Class("ROC", 
#                        inherit = KerasCallback, 
#                        public = list(
#                              losses = NULL,
#                              
#                              on_epoch_end = function(epoch, logs = list()){
#                                    self$losses <- c(self$losses, logs[["loss"]])
#                                    y_pred <- self$model$predict(self$validation_data[[1]], steps = self$params$batch_size)
#                                    score = AUC::roc(labels = self$validation_data[[2]], predictions = y_pred)
#                                    cat(paste("ROC_AUC for epoch", epoch, "=", score))
#                                    
#                              }
#                              
#                        ))

######################
max_features = 30000
embd_size = 200
maxlen = 150
# GLOVE_DIR <- 'Data/embeddings/glove.6B/'
# glove_file <- 'glove.6B.200d.txt'
# GLOVE_DIR <- 'Data/embeddings/glove.6B/'
#glove_file <- 'glove.twitter.27B.200d.txt'
# 
# glove_matrix <- paste0(GLOVE_DIR,glove_file)

cat("Reading data...\n")
# raw_data <- read.csv(file = "new_train_v2.csv", stringsAsFactors = F, as.is = T)
# test_data <- read.csv(file = "new_test_v2.csv", stringsAsFactors = F, as.is = T)
label = c('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
# 
# train_text <- data.frame(comments = raw_data$comment_text, stringsAsFactors = F)
# test_text <- data.frame(comments = test_data$comment_text, stringsAsFactors = F)
# comments <- rbind(train_text, test_text)
# 
# ind <- sample(1:nrow(train_text), size = 20000, replace = F)
# valid_text <- data.frame(comments = train_text[ind, ], stringsAsFactors = F)
# train_text <- data.frame(comments = train_text[-ind, ], stringsAsFactors = F)
# 
# cat("Tokenizing data...\n")
# tokenizer <- text_tokenizer(num_words = max_features)
# tokenizer <- fit_text_tokenizer(tokenizer, x = comments$comments)
# save_text_tokenizer(tokenizer, paste0("saves/tuning_keras_text_tokenizer"))
# 
# X_train <- texts_to_sequences(tokenizer, train_text$comments)
# X_train <-  pad_sequences(X_train, maxlen = maxlen, padding = "post", value = 0, truncating = "post")
# 
# write.csv(X_train, "Tuning_Data/X_train.csv", row.names = F)
# x_valid <- texts_to_sequences(tokenizer, valid_text$comments)
# x_valid <-  pad_sequences(x_valid, maxlen = maxlen, padding = "post", value = 0, truncating = "post")
# write.csv(x_valid, "Tuning_Data/X_valid.csv", row.names = F)
# 
# 
# X_test <- texts_to_sequences(tokenizer, test_text$comments)
# X_test <- pad_sequences(X_test, maxlen = maxlen, padding = "post", value = max_features+2, truncating = "post")
# write.csv(X_test, "Tuning_Data/X_test.csv", row.names = F)
# 
# Y_train <- as.matrix(raw_data[-ind,label])
# y_valid <- as.matrix(raw_data[ind, label])
# 
# write.csv(Y_train, "Tuning_Data/Y_train.csv", row.names = F)
# write.csv(y_valid, "Tuning_Data/y_valid.csv", row.names = F)
X_train <- as.matrix(read.csv("Tuning_Data/X_train.csv"))
x_valid <- as.matrix(read.csv("Tuning_Data/X_valid.csv"))
X_test <- as.matrix(read.csv("Tuning_Data/X_test.csv"))

Y_train <- as.matrix(read.csv("Tuning_Data/Y_train.csv"))
y_valid <- as.matrix(read.csv("Tuning_Data/y_valid.csv"))

tokenizer <- load_text_tokenizer("Tuning_Data/tuning_keras_text_tokenizer")
cat("Dataset is ready!\n")

# create_embedding_matrix <- function(){
#       
#       cat("Indexing words vectors...\n")
#       i=0
#       embeddings_index <- new.env(parent = emptyenv())
#       cat("Reading lines from glove...")
#       lines <- readLines(file.path(glove_matrix))
#       #lines <- readLines(file.path('Data/embeddings/wiki-news-300d-1M-subword.vec'))
#       
#       for (line in lines) {
#             values <- strsplit(line, ' ',fixed = T)[[1]]
#             word <- values[[1]]
#             if (i %% 5000 == 0) {
#                   cat(paste(word, ":",i+1))
#             }
#                   
#             embeddings_index[[word]] <- as.numeric(values[-1])
#             i = i+1
#       }
#       
#       saveRDS(embeddings_index, paste0(glove_matrix,".RDS"))
#       
#       cat(sprintf('Found %s word vectors.\n', length(embeddings_index)))
#       word_index <- tokenizer$word_index
#       cat('Preparing embedding matrix.\n')
#       rm(lines)
#       
#       # prepare embedding matrix
#       num_words <- min(max_features, length(word_index))
#       prepare_embedding_matrix <- function() {
#             embedding_matrix <- matrix(0L, nrow = num_words, ncol = embd_size)
#             for (word in names(word_index)) {
#                   index <- word_index[[word]]
#                   if (index < max_features)
#                         embedding_vector <- embeddings_index[[word]]
#                   if (!is.null(embedding_vector)) {
#                         # words not found in embedding index will be all-zeros.
#                         embedding_matrix[index+1,] <- embedding_vector
#                   }
#             }
#             write.csv(embedding_matrix, paste0(GLOVE_DIR,"glovetwitter27B200d","_embedding_matrix.csv"), row.names = F)
#             return(embedding_matrix)
#       }
#       
#       embedding_matrix <- prepare_embedding_matrix()
#       return(embedding_matrix)
# }
# 
# system.time(embedding_matrix <- create_embedding_matrix())
embedding_matrix <- as.matrix(read.csv("Tuning_Data/glovetwitter27B200d_embedding_matrix.csv"))

filter_size <- c(1, 2, 3, 5)
num_filters = 32

textCNN_model <- function(){
      
      main_input = layer_input(shape = c(maxlen))
      
      input = main_input %>% 
            layer_embedding(input_dim = max_features, output_dim = embd_size, weights = list(embedding_matrix), trainable = F) %>%
            layer_spatial_dropout_1d(rate = 0.4) %>%
            layer_reshape(target_shape = c(maxlen, embd_size, 1))
      
      conv1 <- input %>% 
            layer_conv_2d(filters = num_filters, kernel_size = c(filter_size[1], embd_size), activation = "elu", kernel_initializer = "normal") %>%
            layer_max_pooling_2d(pool_size = c(maxlen - filter_size[1] + 1, 1))
      
      conv2 <- input %>% 
            layer_conv_2d(filters = num_filters, kernel_size = c(filter_size[2], embd_size), activation = "elu", kernel_initializer = "normal") %>%
            layer_max_pooling_2d(pool_size = c(maxlen - filter_size[2] + 1, 1))
      
      conv3 <- input %>% 
            layer_conv_2d(filters = num_filters, kernel_size = c(filter_size[3], embd_size), activation = "elu", kernel_initializer = "normal") %>%
            layer_max_pooling_2d(pool_size = c(maxlen - filter_size[3] + 1, 1))
      
      conv4 <- input %>% 
            layer_conv_2d(filters = num_filters, kernel_size = c(filter_size[4], embd_size), activation = "elu", kernel_initializer = "normal") %>%
            layer_max_pooling_2d(pool_size = c(maxlen - filter_size[4] + 1, 1))
      
      combine = layer_concatenate(inputs = list(conv1, conv2, conv3, conv4), axis = 1) %>%
            layer_flatten() %>% 
            layer_dropout(rate = 0.2)
      
      output = combine %>% 
            layer_dense(units = 6, activation = "sigmoid")
      
      model <- keras_model(inputs = main_input, outputs = output)
      
      model %>% compile(loss = "binary_crossentropy",
                        optimizer = optimizer_adam(lr = 0.003), 
                        metrics = "accuracy")
      
return(model)
}

model1 <- textCNN_model()

history <- model1 %>% fit(x = X_train, y = Y_train, 
                          batch_size = 64, epochs = 3,
                          validation_data = list(x_valid, y_valid), 
                          callbacks = list(callback_early_stopping(patience = 3, mode = "max")))

cat("Calculating AUC on validation set...")
valid_preds <- model1 %>% predict(x_valid)

cat("\nAUC on validation set:", Metrics::auc(predicted = valid_preds, actual = y_valid))

cat("\nSaving model...")
current_time <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
save_model_hdf5(model1, paste0("Tuning_Data/saves/", current_time, "_model_cpu.hdf5"), include_optimizer = T)
save_model_weights_hdf5(model1, paste0("Tuning_Data/saves/", current_time, "_model_weights_cpu.hdf5"))
#save_text_tokenizer(tokenizer, paste0("saves/keras_text_tokenizer", current_time))
cat("\nDone!")
# cat("\nPredicting test data...")
# system.time(test_preds <- as.data.frame(predict(model1, X_test)))
# colnames(test_preds) <- label
# submissions <- read.csv("Data/sample_submission.csv", stringsAsFactors = F)
# test_preds$id <- submissions$id
# #  
# cat("\nWriting csv file!")
# write.csv(test_preds, paste0("results/textCNN_",format(Sys.time(), "%Y_%m_%d_%H_%M_%S"),".csv"), row.names = F)
