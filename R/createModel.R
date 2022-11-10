
#' Title
#'
#' @param data data.frame with all features and labels
#' @param labels vector of matching labels
#'
#' @return
#' @export createModel
#' @importFrom dplyr %>%
#'
#' @examples
createModel <- function(data, labels, verbose=TRUE) {
  data=spam.raw
  labels=as.factor(spam.raw$label)
  set.seed(32984) # reproducibility
  indices <- caret::createDataPartition(labels, times = 1, p = 0.7, list = FALSE)
  train <- data[indices, ]
  test <- data[-indices, ]
  if(verbose) print("Tokenizing...")
  train.dfm <- quanteda::tokens(x = train$text, what = "word", remove_punct = TRUE,
                                   remove_symbols = TRUE, remove_numbers = TRUE, split_hyphens = TRUE) %>%
    quanteda::tokens_tolower() %>%
    quanteda::tokens_remove(quanteda::stopwords("en")) %>%
    quanteda::tokens_wordstem(language = "en") %>%
  # "But lemmatizing needs to be done with tokens_replace. Note the difference
  # between the 2, in lemmatizing "am" is changed into "be" as this is the
  # lemma. In the lexicon package there is a table called hash_lemmas that you
  # can use as a dictionary. There is no default lemma function in quanteda."
    quanteda::dfm(tolower = FALSE) %>%
    quanteda::convert(to = "data.frame") %>%
    cbind(Label = train$label, .) %>%
    dplyr::select(-doc_id)

  # colnames(train.dfm)[c(138, 140, 209, 212)]  # not technically legal names for a dataframe, e.g. "try:wal"
  colnames(train.dfm) <- make.names(colnames(train.dfm))

  # set up 3-repeated 10-fold cross-validation
  set.seed(48743)
  cv.folds <- caret::createMultiFolds(y= train$label, k = 10, times = 3)
  cv.ctrl <- caret::trainControl(method = "repeatedcv", number = 10, savePredictions = 'all',
                                 repeats = 3, index = cv.folds, allowParallel = TRUE)
  # cv.ctrl <- caret::trainControl(method="cv", allowParallel = TRUE)
  cv.ctrl <- caret::trainControl(method = "boot", number = 5, allowParallel = TRUE)


  # library(doMC); registerDoMC(cores=3)
  library(doParallel);cl <- makePSOCKcluster(14);registerDoParallel(cl)
  if(verbose) print("Finding near-zero variance predictors...")
  nzv2 <- caret::nearZeroVar(train.dfm, freqCut = 200, foreach=TRUE, allowParallel = TRUE)
  start.time <- Sys.time()
  # cls <- snow::makeCluster(spec = 3, type = "SOCK")
  # doSNOW::registerDoSNOW(cls)
  # cl <- doParallel::makePSOCKcluster(5)
  # registerDoParallel(cl)

  if(verbose) print("Training model...")
  # Make model: single decision tree to begin with
  # model.logreg <- caret::train(Label ~ ., data = train.dfm[, 1:10], method='glm', trControl = cv.ctrl, family='binomial')#, tuneLength=7)
  # m <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='rpart', tuneLength=7);m
  # m2 <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='glm', family="binomial", tuneLength=7);m2
  # m3 <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='glmnet', family="binomial",
  #                    type.logistic="modified.Newton", tuneLength=7);m3

  m <- caret::train(train.dfm[, -c(1, nzv2)], train.dfm[,1], trControl=cv.ctrl, method='lda', tuneLength=7)
  # m5 <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='lda2', tuneLength=7);m5
  # # m6 <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='stepLDA', tuneLength=7);m6
  # # m7 <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='nbDiscrete', tuneLength=7);m7
  # # m8 <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='nb', tuneLength=7);m8

  # registerDoSEQ()
  stopCluster(cl)

  # print("got to here")
  # snow::stopCluster()
  # stopCluster(cl)
  total.time <- Sys.time() - start.time
  print(paste("Elapsed time:", total.time))
  m
  }
