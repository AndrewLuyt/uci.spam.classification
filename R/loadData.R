#' Return the supplied raw spam dataset as a data.frame
#'
#' @return NULL
#' @export loadData
#'
loadData <- function() {
  # The supplied data set is UTF-16 encoded and will report "embedded nulls" errors if we don't
  # specify the encoding. Apparently UFT-16 is a common format for windows-only tools and tends
  # to mess up in other environments.
  if (file.exists("./inst/extdata/spam.raw.rds")) {
    spam.raw <- readRDS("inst/extdata/spam.raw.rds")
  } else {
    spam.raw <- readr::read_csv("./inst/extdata/spam.csv", col_names = c('label', 'text'), col_select = c("label", "text"),
                                col_types = c("f", "c"), skip = 1, locale = readr::locale(encoding="UTF-16LE"))
    if ( length(which(!stats::complete.cases(spam.raw))) != 0)
      stop("Incomplete cases detected in data source, fix before continuing.")
    spam.raw <- createFeatures(spam.raw)
  }
  spam.raw
}

labelProportions <- function(label) {
  prop.table(table(label))
}

createFeatures <- function(dataset) {
  dataset$text.length <- nchar(dataset$text)
  dataset
}
