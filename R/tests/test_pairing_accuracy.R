source("R/pairing_accuracy.R", chdir = TRUE)
library(data.table)
library(dplyr)
library(testthat)


#load test data and expected results files 
actual_pairings_test_file <- "R/tests/test_mannual_map_ref_data.csv"
scored_pairings_test_file <- "R/tests/test_var_similarity_scores_rank_data.csv"
expected_predictions_file <- "R/tests/test_predictions_data.csv"
expected_accuracy_file <- "R/tests/test_pairings_accuracy_data.csv"


test_data <- as_tibble(fread(scored_pairings_test_file, header = T, sep = ',', stringsAsFactors=FALSE))
expected_actual_pairings <- as_tibble(fread(actual_pairings_test_file, header = T, sep = ',', stringsAsFactors=FALSE)) 
expected_accuracy <- as_tibble(fread(expected_accuracy_file, header = T, sep = ',', stringsAsFactors=FALSE))

#colnames for parameters inputs
ref_id_col <- "refID"
class_col <- "concept"
ref_group_col <-  "ref_group"
paired_id_col <- "pairID" 
true_pairing_col <- "true_pairing" 
paired_group_col <- "pair_group"

#check concept and ref ID for test/expected data are equal 
test_that("equal ref ID's in test datasets", {
    expect_equal(sort(unique(test_data[[ref_id_col]])),
                 sort(unique(expected_actual_pairings[[ref_id_col]])))
})

test_that("equal concepts in test and actual_pairing datasets", {
    expect_equal(sort(unique(test_data[[class_col]])),
                 sort(unique(expected_actual_pairings[[class_col]])))       
})

#parameters for pairing accuracy functions
actual_params <- list("ref_ID_col" = ref_id_col,
                        "class_col" = class_col,
                        "ref_group_col" = ref_group_col,
                        "multiple_correction" = F)

function(data, ref_ID_col, paired_ID_col, true_pairing_col, rank_col, rank_cutoffs = c(1,5,10), class_col = NULL, ref_group_col = NULL, multiple_correction = F, paired_group_col = NULL, remove_refID_stats = T) 
accuracy_params <- list("ref_ID_col" = ref_id_col, 
                        "paired_ID_col" = paired_id_col, 
                        "true_pairing_col" = true_pairing_col, 
                        "rank_col" = "rank",
                        "rank_cutoffs" = c(1,5,10), 
                        "class_col" = class_col, 
                        "ref_group_col" = ref_group_col,
                        "multiple_correction" = T,
                        "paired_group_col" = paired_group_col,
                        "remove_refID_stats" = F)

#function to calculate and sort results for function passed in. 
calculate_results <- function(test_data, test_function, params, expected_cols) {
    sort_cols <- unlist(params[grep("^ref|^class", names(params))], use.names = F)
    do.call(test_function , 
            c(list("data" = as_tibble(test_data)), params)) %>% 
        select(all_of(expected_cols)) %>% 
        unique() %>%
        arrange(across(sort_cols))
}

#function to select the right columns apply filters to expected data based on different parameter inputs to pairing accuray function calls. 
extract_expected_for_params <- function(expected_data, params, col_regex = "^n_|concept|^ref|pred|^actual|precision|recall|F1") {
    suffix <- if(params[["multiple_correction"]]) "_set" else "_var"
    
    #set columns that will be in output results data 
    expected_cols <- gsub("_set$|_var$", "", grep(col_regex,  names(expected_data), value = T))
    sort_cols <- unlist(params[grep("^class|^ref", names(params))], use.names = F)
    
    #filter rows if class_col/ref_group parameters are NULL
    filter_concepts <- unique(expected_data$concept)
    if(is.null(params[["class_col"]])) { 
        filter_concepts <- "Age (years)"
    }
    if(is.null(params[["ref_group_col"]]) & !is.null(params[["class_col"]])) {
        filter_concepts <- c("CVD", "Death")
    }
    
    #filter/select colums in expected data 
    expected_data %>% 
        rename_with(~gsub(suffix, "", .x), ends_with(suffix)) %>%  
        select(all_of(unique(expected_cols))) %>% 
        filter(concept %in% filter_concepts) %>% 
        arrange(across(sort_cols))
}
  

test_that("test summarize_actual_pairings", {
    actual_params[["multiple_correction"]] <- F 
    
    expected <- extract_expected_for_params(expected_actual_pairings, actual_params)
    results <- calculate_results(test_data, "summarize_actual_pairings", actual_params, colnames(expected)) 
    
    expect_equal(results, expected, check.attributes = F)
})

test_that("test summarize_actual_pairings multiple correction", {
    actual_params[["multiple_correction"]] <- T

    expected <- extract_expected_for_params(expected_actual_pairings, actual_params)
    results <- calculate_results(test_data, "summarize_actual_pairings", actual_params, colnames(expected)) 
    
    expect_equal(results, expected, check.attributes = F)
})


test_that("test calc_accuracy", {
    actual_params[["multiple_correction"]] <- F
    
    expected <- extract_expected_for_params(expected_accuracy, accuracy_params)
    results <- calculate_results(test_data, "calc_accuracy", accuracy_params, colnames(expected)) 
    
    expect_equal(results, expected, check.attributes = F)
})


test_that("test calc_accuracy multiple correction", {
    actual_params[["multiple_correction"]] <- T
    
    expected <- extract_expected_for_params(expected_accuracy, accuracy_params)
    results <- calculate_results(test_data, "calc_accuracy", accuracy_params, colnames(expected)) 
    
    expect_equal(results, expected, check.attributes = F)
})

#check if pairing order changes score
combo_match <- test_data %>% group_by(concept) %>% 
    filter(pairID %in% refID & refID %in% pairID) %>% 
    select(all_of(c("refID" = ref_id_col , "pairID" = paired_id_col)), concept, matches("^score")) %>%
    mutate(op_pair_idx = match(paste0(pairID, refID), paste0(refID, pairID)), 
           op_pair_score = score[op_pair_idx]) 

all.equal(combo_match$score, combo_match$op_pair_score)

##~~~~~~~~~~~~~~~~~~~toy pairing set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
# char_overlap <- function(str1,str2) {
#     if(str1 == str2) {return(1)}
#     chr1 <- strsplit(gsub(" ", "",str1), "")[[1]]
#     chr2 <- strsplit(gsub(" ", "",str1), "")[[1]]
#     sum(sapply(chr1, function(x){ x %in% chr2}))/max(nchar(str1), nchar(str2))
# }
# 
# addTaskCallback(function(...) {set.seed(3);TRUE})
# 
# toy_data <- as_tibble(
#     list(
#         "concept" = gsub(" .*$", "", row.names(mtcars)),
#         "ref_group" = paste0("cyl=", mtcars$cyl, "-gear=", mtcars$gear),
#         "refID" = row.names(mtcars))) %>%
#     group_by(concept) %>%
#     filter(length(concept) >= 2 | concept == "AMC") %>% 
#     ungroup() %>%
#     mutate(pairID = list(refID), 
#            pair_group = list(ref_group)) %>% 
#     unnest(c(pairID, pair_group)) %>% 
#     filter(refID != pairID) %>%
#     mutate(score = map2_dbl(refID, pairID, ~char_overlap(.x,.y))) %>% 
#     arrange(refID, score) %>% 
#     group_by(refID, pair_group) %>%
#     mutate(rank = dense_rank(score), 
#            rank = replace(rank, score == 0, NA), 
#            true_pairing = map2_lgl(concept, pairID, ~grepl(.x,.y))) 




