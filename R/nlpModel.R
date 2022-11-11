
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
  # https://www.r-bloggers.com/2019/08/no-visible-binding-for-global-variable/
  doc_id <- NULL # Remove R CMD CHECK note. See URL above.
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

    quanteda::dfm(tolower = FALSE) %>%
    quanteda::convert(to = "data.frame") %>%
    cbind(Label = train$label, .data) %>%
    dplyr::select(-doc_id)

  # some tokens will create illegal/problematic column names for a data frame
  colnames(train.dfm) <- make.names(colnames(train.dfm))

  set.seed(48743)

  # 3-repeated 10-fold cross-validation
  # cv.folds <- caret::createMultiFolds(y= train$label, k = 10, times = 3)
  # cv.ctrl <- caret::trainControl(method = "repeatedcv", number = 10, savePredictions = 'all',
  #                                repeats = 3, index = cv.folds, allowParallel = TRUE)
  # set up bootstrap resampling
  cv.ctrl <- caret::trainControl(method = "boot632", number = 7, allowParallel = TRUE)

  cl <- parallel::makePSOCKcluster(3)
  doParallel::registerDoParallel(cl)

    if(verbose) print("Finding near-zero variance predictors...")
  nzv2 <- caret::nearZeroVar(train.dfm, freqCut = 200, foreach=TRUE, allowParallel = TRUE)

  if (verbose) start.time <- Sys.time()
  if(verbose) print("Training model...")
  # m <- caret::train(Label ~ ., data = train.dfm[, -nzv2], trControl=cv.ctrl, method='rpart', tuneLength=7);m
  m <- caret::train(train.dfm[, -c(1, nzv2)], train.dfm[,1], trControl=cv.ctrl, method='lda')
  parallel::stopCluster(cl)

  if (verbose) {
    total.time <- Sys.time() - start.time
    print(paste("Elapsed time:", total.time))
    print(caret::postResample(stats::predict(m), train.dfm$Label))
  }
  m
}
