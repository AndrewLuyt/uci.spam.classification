
#' Create email classifier with Linear Discriminant Analysis
#'
#' @param data A data.frame with all features and labels
#' @param labels A vector of matching labels
#' @param verbose Print progress messages
#'
#' @return A caret model that predicts spam or ham
#' @export nlpModel
#' @importFrom dplyr %>%
#' @importFrom rlang .data
#'
nlpModel <- function(data, labels, verbose=TRUE) {
  set.seed(1712) # reproducibility
  indices <- caret::createDataPartition(labels, times = 1, p = 0.7, list = FALSE)
  train <- data[indices, ]
  test <- data[-indices, ]
  if(verbose) print("Tokenizing...")
  train.dfm <- quanteda::tokens(x = train$text, what = "word", remove_punct = TRUE,
                                   remove_symbols = TRUE, remove_numbers = TRUE, split_hyphens = TRUE) %>%
    quanteda::tokens_tolower() %>%
    quanteda::tokens_remove(quanteda::stopwords("en")) %>%
    quanteda::tokens_wordstem(language = "en") %>%

    quanteda::dfm(tolower = FALSE) %>%
    quanteda::convert(to = "data.frame") %>%
    dplyr::select(-"doc_id")
  train.dfm <- cbind(Label = train$label, train.dfm)

  # some tokens will create illegal/problematic column names for a data frame
  colnames(train.dfm) <- make.names(colnames(train.dfm))

  set.seed(1712)

  # 3-repeated 10-fold cross-validation
  # cv.folds <- caret::createMultiFolds(y= train$label, k = 10, times = 3)
  # cv.ctrl <- caret::trainControl(method = "repeatedcv", number = 10, savePredictions = 'all',
  #                                repeats = 3, index = cv.folds, allowParallel = TRUE)
  # 10-fold cross-validation
  # cv.folds @<- caret::createFolds(y=train$label, k=10)
  cv.ctrl <- caret::trainControl(method = "cv", number = 10, allowParallel = TRUE)
  # set up bootstrap resampling
  # cv.ctrl <- caret::trainControl(method = "boot", number = 7, allowParallel = TRUE)

  doParallel::registerDoParallel(cl=3, cores=3)

  if(verbose) print("Finding near-zero variance predictors...")
  nzv2 <- caret::nearZeroVar(train.dfm, freqCut = 200, foreach=TRUE, allowParallel = TRUE)

  if (verbose) start.time <- Sys.time()
  if (verbose) print("Training model...")
  # m <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='rpart', tuneLength=7);m
  m <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='lda')
  doParallel::stopImplicitCluster()

  if (verbose) {
    total.time <- Sys.time() - start.time
    print(paste("Elapsed time:", total.time))
    print(caret::postResample(stats::predict(m), train.dfm$Label))
  }
  m
}
