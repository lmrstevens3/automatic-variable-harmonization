library(tidyverse)

reduce_join <- function(start_data, join_data, data_cols = NULL, join_type = c("left", "full")[1], ...) { 
    #' Function to join data from multiple (large) files. 
    #' @param start_file: first dataframe to be loaded for reduce join function
    #' @param scores_files: List of file paths for additional data to be joined. 
    #' Data columns not in id_cols are concatenated with the name of the file in scores_scores files if named and the index if not named.
    #' @param file_cols: list of vectors of columns names to use for each file. If vectors are named, columns will be renamed using the name. 
    #' The list should equal the length of start_file and join_file combined, if not, it will not be used. 
    #' @param join_type: type of join to perform-left of full. Default is left. 
    #' @...: additional parameters passed to Left_join/full_join
    if(length(data_cols) > 0 & length(data_cols) != length(join_data) + 1) {
        warning("length of data_cols not equal length to data input and will not be used")
    }
    read_data_if_needed(start_data) %>% 
        {if(length(data_cols) == length(join_data) + 1) {
             reduce2(c(list(select(., all_of(data_cols[[1]]))), join_data), 
                     data_cols[c(2:length(data_cols))], 
                     load_join_data, 
                     join_type = join_type, ...)
        } else {
            reduce(c(list(.), join_data), 
                   load_join_data, 
                   join_type = join_type, ...)
        }}
}

read_data_if_needed <- function(x) { 
    if((!is.data.frame(x)) & is.character(x) & length(x) == 1)  {
        if(file.exists(x)) {
            read_csv(x) 
        } else stop("items in join_data must be a dataframe or exisiting file") 
    } else {
        x 
    } 
}

load_join_data <- function(x, y, y_cols = NULL, x_cols = NULL, join_type = c("left", "full")[1], ...) { 
    #' wrapper to join data and rename cols if necessary- data will be loaded if a file path is passed in
    join_data <- list(x, y) 
    join_data <- lapply(c(1:2), function(x) {
        suppressMessages(read_data_if_needed(join_data[[x]]))
    })
    
    if(length(x_cols) > 0) {
        join_data[[1]] <-select(join_data[[1]], all_of(x_cols))
    }
    if(length(y_cols) > 0) {
        join_data[[2]] <-select(join_data[[2]], all_of(y_cols))
    }
    
    if(join_type == "full") { 
        full_join(join_data[[1]], join_data[[2]] , ...) 
    } else {
        left_join(join_data[[1]], join_data[[2]], ...) 
    }
}


test_reduce_join <- function() { 
    #should result in dataframe with 1 species columns, 3 score columns and 21 rows
    tst_data <- list(head(iris %>% filter(Species == "setosa") %>% select(Species, score1 = Sepal.Width)) %>%
                         mutate(Species = paste0(Species, row_number())),
                     head(iris %>% filter(Species == "setosa") %>% select(Species, score2 = Sepal.Length), 10) %>%
                         mutate(Species = paste0(Species, row_number())),
                     head(iris %>% filter(Species == "setosa") %>% select(Species, score3 = Sepal.Width), 5) %>%
                         mutate(Species = paste0(Species, row_number()))) 
    tst_data2 <- list(head(iris %>% filter(Species == "setosa") %>% select(Species, score = Sepal.Width)) %>%
                         mutate(Species = paste0(Species, row_number())),
                     head(iris %>% filter(Species == "setosa") %>% select(Species, score = Sepal.Length), 10) %>%
                         mutate(Species = paste0(Species, row_number())),
                     head(iris %>% filter(Species == "setosa") %>% select(Species, score = Sepal.Width), 5) %>%
                         mutate(Species = paste0(Species, row_number())))
    # sapply(c(1:length(tst_data2)), function(x) {
    #     write_csv(tst_data2[[x]], paste0("R/tests/iris_example_scores", x, ".csv"))})
    tst_data3 <- list("R/tests/iris_example_scores1.csv",
                      "R/tests/iris_example_scores2.csv",
                      "R/tests/iris_example_scores3.csv")
    tst_cols <- list(c("id" = "Species", "score1" = "score"),
                     c("id" = "Species", "score2" = "score"), 
                     c("id" = "Species", "score3" = "score"))
    
    #expected results
    exp_load_join_left <- as_tibble(list("Species" = c(paste0("setosa", 1:6)),
                                         "score1" = c(iris$Sepal.Width[1:6]),
                                         "score2" = c(iris$Sepal.Length[1:6])))
                               
    exp_load_join_full <- as_tibble(list("Species" = c(paste0("setosa", 1:10)),
                                        "score1" = c(iris$Sepal.Width[1:6], rep(NA,4)),
                                        "score2" = iris$Sepal.Length[1:10]))
                               
    exp_left_join <- as_tibble(list("id" = c(paste0("setosa", 1:6)),
                                    "score1" = iris$Sepal.Width[1:6],
                                    "score2" = iris$Sepal.Length[1:6],
                                    "score3" = c(iris$Sepal.Width[1:5], NA)))
                                  
    exp_full_join <-  as_tibble(list("id" = c(paste0("setosa", 1:10)), 
                                     "score1" = c(iris$Sepal.Width[1:6],rep(NA,4)),
                                     "score2" = iris$Sepal.Length[1:10],
                                     "score3" = c(iris$Sepal.Width[1:5], rep(NA,5))))
    #load_join test results
    res_tst1_left <- load_join_data(tst_data[[1]], tst_data[[2]])
    res_tst2_full <- load_join_data(tst_data[[1]], tst_data[[2]], join_type = "full")
    
    res_tst3_left <- load_join_data(tst_data2[[1]], tst_data2[[2]], 
                                    y_cols = c("Species", "score2" = "score")) %>%
        rename("score1" = "score")
    res_tst4_full <- load_join_data(tst_data2[[1]], tst_data2[[2]], 
                                    y_cols = c("Species", "score2" = "score"), 
                                    x_cols = c("Species", "score1" = "score"),
                                    join_type = "full")
    
    res_tst5_left <- load_join_data(tst_data3[[1]], tst_data3[[2]], y_cols = tst_cols[[2]], x_cols = tst_cols[[1]]) 

    
    load_join_tests <- c(isTRUE(all_equal(res_tst1_left, exp_load_join_left)),
                         isTRUE(all_equal(res_tst2_full, exp_load_join_full)),
                         isTRUE(all_equal(res_tst3_left, exp_load_join_left)),
                         isTRUE(all_equal(res_tst4_full, exp_load_join_full)),
                         isTRUE(all_equal(res_tst5_left, 
                                          exp_load_join_left %>% rename("id" = "Species"))))
                            
    
    if(sum(load_join_tests) != length(load_join_tests)) {
        stop("Tests Failed: load_join- ", 
             paste0(which(load_join_tests %in% F), collapse = ", "), " tests failed")
    }
    
    #reduce_join test results
    res2_tst1_left <-reduce_join(tst_data[[1]], tst_data[c(2:3)])
    res2_tst2_left <-reduce_join(tst_data2[[1]], tst_data2[c(2:3)], tst_cols)
    res2_tst3_full <-reduce_join(tst_data3[[1]], tst_data3[c(2:3)], tst_cols, join_type = "full")

    reduce_join_tests <- c(isTRUE(all_equal(res2_tst1_left, exp_left_join %>% rename(Species = id))),
                           isTRUE(all_equal(res2_tst2_left, exp_left_join)),
                           isTRUE(all_equal(res2_tst3_full, exp_full_join)))
                         
                            
    if(sum(reduce_join_tests) != length(reduce_join_tests)) {
        stop("Tests Failed: reduce_join- ", 
             paste0(which(reduce_join_tests %in% F), collapse = ", "), " tests failed")
    }
    message("Tests Passed! :)")
}

#test_reduce_join()

extract_pairable_pairs <- function(data, ref_ID_col, paired_ID_col, ref_group_col = NULL, paired_group_col = NULL, class_col = NULL, classes_to_remove = NULL, keep_single_ref_class = F) {
    #' @param data: data frame of pairings with a column of ranks. 
    #' @param ref_ID_col: the column in data that contains ref IDs.
    #' @param paired_ID_col: the column in data that contains pairing IDs.
    #' @param ref_group_col: the column in data that contains groups for ref IDs. If supplied with paired_group_col, pairs from the same group will be filtered out
    #' @param paired_group_col: the column in data that contains groups for the paired IDs. If supplied with ref_group_col, pairs with the same ID will be filtered out
    #' @param class_col: the column in data that contains classes for the ref/paired IDs. If supplied, pairs from the same group will be filtered out by class 
    #' @param classes_to_remove: values to remove pairings of a certain class
    #' @param keep_single_ref_class: if TRUE, do not remove pairings in classes with a single refID (no pairings of IDs from the same class exist) 
    data %>% dplyr::filter(!!as.name(ref_ID_col) != !!as.name(paired_ID_col)) %>%
    {if(length(c(ref_group_col, paired_group_col)) == 2){
        dplyr::group_by(., across(any_of(class_col))) %>%
            dplyr::filter(!!as.name(ref_group_col) != !!as.name(paired_group_col)) %>% 
            ungroup()
     } else {.}}  %>%
    {if(length(class_col) > 0 ) { #remove classes with only one refID
        dplyr::filter(., if_any(any_of(class_col), ~(!.x %in% classes_to_remove))) %>%
            {if(keep_single_ref_class) {.}
            else {
                dplyr::group_by(., across(all_of(class_col))) %>%
                    dplyr::filter(., length(unique(!!as.name(ref_ID_col))) > 1)
                }}
    } else {.}}
}

rank_scores <- function(data, score_col, refID_col, rank_by_col = NULL, filter_ranks = F, rank_cutoff = 10){
    #' sorts data by score_col (descending) and dense ranks pairings with highest similarity scores. 
    #' Pairings with a score of zero receive a rank value of NA. A rank_<score_col> column is appended to data 
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param score_col: the column name of the column in data containing similarity scores
    #' @param ref_ID_col: the column in data that contains unique ref IDs
    #' @rank_by_col: An optional grouping variable that defines if ranking 1:n should be done by group instead of for all pairings for a given ref ID. 
    #' For example, if scores for a ref come from multiple sources, the column listing the scores source could be supplied, 
    #' and ref ID would be ranked 1:n scores for each source instead of 1:n scores in all sources
    data %>% group_by(across(any_of(c(refID_col, rank_by_col)), add = T)) %>% 
        arrange(across(all_of(score_col), ~desc(.x)), .by_group = T) %>% 
        mutate(across(all_of(score_col), ~replace(dense_rank(desc(.x)), .x == 0, NA), .names = "{.col}_rank")) %>%
        select(grep(paste0(score_col,"_rank$"), colnames(.), invert = T), matches(score_col)) %>%
        {if(filter_ranks) {filter(., if_all(paste0(score_col,"_rank"), ~.x <= rank_cutoff))} else {.}}
}

calc_ranks_all_scores <- function(data, score_variables, ref_ID_col, rank_by_col = NULL, filter_all_ranks = F, rank_cutoff = 10){
    #wrapper function to parallelize or append a set of rank cols using rank_scores function for multiple pairing scores
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param score_variables: vector of the score column names of the columns in data containing similarity scores to be ranked
    #' @param ref_ID_col: the column in data that contains unique ref IDs
    #' @rank_by_col: An optional grouping variable that defines if ranking 1:n should be done by group instead of for all pairings for a given ref ID. 
    #' see rank_scores for additional details
    #' @parallel: if True, code will be parallelized for optmization 
    #' @cores: number of cores to use for parallel processing (default is cores available - 2)
    
    for(i in score_variables) {
        print(paste0("calculating ", i, " ranks"))
        data <- rank_scores(data, i, ref_ID_col, rank_by_col, filter_ranks = F, rank_cutoff) 
    }
    if(filter_all_ranks) {
        data <- data %>% ungroup() %>% 
            mutate(rank_cols_min = pmap_dbl(select(., matches("_rank$")), pmin, na.rm =T)) %>% 
            filter(!is.na(rank_cols_min) & rank_cols_min <= rank_cutoff) %>% 
            select(-rank_cols_min)
    }
    return(data)
}






    