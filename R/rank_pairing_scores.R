library(tidyverse)

extract_pairable_pairs <- function(data, ref_ID_col, paired_ID_col, ref_group_col = NULL, paired_group_col = NULL) {
    #' @param data: data frame of pairings with a column of ranks. 
    #' @param ref_ID_col: the column in data that contains ref IDs.
    #' @param paired_ID_col: the column in data that contains pairing IDs.
    #' @param ref_group_col: the column in data that contains groups for ref IDs. If supplied, pairs from the same group will be filtered out
    #' @param paired_ID_col: the column in data that contains groups for the paired IDs. If supplied, pairs from the same group will be filtered out
    
    data %>% filter(!!as.name(ref_ID_col) != !!as.name(paired_ID_col)) %>%
    {if(length(c(ref_group_col, paired_group_col)) == 2){
        filter(., !!as.name(ref_group_col) != !!as.name(paired_group_col))
    }}
}

rank_scores <- function(data, score_col, refID_col, rank_by_col = NULL){
    #' sorts data by score_col (descending) and dense ranks pairings with highest similarity scores. 
    #' Pairings with a score of zero recieve a rank value of NA. A rank_<score_col> column is appended to data 
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param score_col: the column name of the column in data containing similarity scores
    #' @param ref_ID_col: the column in data that contains unique ref IDs
    #' @rank_by_col: An optional grouping variable that defines if ranking 1:n should be done by group instead of for all pairings for a given ref ID. 
    #' For example, if scores for a ref come from multiple sources, the column listing the scores source could be supplied, 
    #' and ref ID would be ranked 1:n scores for each source instead of 1:n scores in all sources
    data %>% group_by(across(all_of(c(refID_col, rank_by_col))), add = T) %>% 
        arrange(across(all_of(score_col), ~desc(.x)), .by_group = T)  %>% 
        mutate(across(all_of(score_col), list("rank" = ~dense_rank(desc(.x))))) %>% 
        mutate(across(all_of(paste0(score_col, "_rank")), ~replace(.x, !!as.name(score_col) == 0, NA))) %>%
        dplyr::select(grep(paste0(score_col,"_rank$"), colnames(.), invert = T), matches(score_col))
}

calc_ranks_all_scores <- function(data, score_variables, ref_ID_col, rank_by_col = NULL, filter_all_ranks = T, rank_cutoff = 10, parallel = F, cores = (detectCores() - 2)){
    #wrapper function to parallelize or append a set of rank cols using rank_scores function for multiple pairing scores
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param score_variables: vector of the score column names of the columns in data containing similarity scores to be ranked
    #' @param ref_ID_col: the column in data that contains unique ref IDs
    #' @rank_by_col: An optional grouping variable that defines if ranking 1:n should be done by group instead of for all pairings for a given ref ID. 
    #' see rank_scores for additional details
    #' @parallel: if True, code will be parallelized for optmization 
    #' @cores: number of cores to use for parallel processing (default is cores available - 2)
    if (parallel) {
        data <- data %>% group_by(across(ref_ID_col)) %>% 
            multidplyr::partition(., new_cluster(cores))
    }
    for(i in score_variables) {
        print(paste0("calculating ", i, " ranks"))
        data <- rank_scores(data, i, ref_ID_col, rank_by_col)
    }
    if(filter_all_ranks) {
        data <- data %>% ungroup() %>% 
            mutate(rank_cols_min = pmap_dbl(select(., matches("_rank$")), pmin, na.rm =T)) %>% 
            filter(rank_cols_min <= rank_cutoff) %>% 
            select(-rank_cols_min)
    }
    if(parallel) {
        data <- data %>% multidplyr::collect(.)
    }
    return(data)
}






    