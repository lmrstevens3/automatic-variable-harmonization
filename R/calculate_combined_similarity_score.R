library(dplyr)
library(readr)

#Laura Stevens
#project directory: ~/Dropbox/Graduate School/Data Integration and Harmonization/variable_mapping_automated


overall_similarity <- function(data, score_cols){
    #' Calculates combined score for score_cols columns in data. 
    #' If 2 additional columns are provided (length of score_cols is 3) combined score is: 1-((1-score1)^(1+score2^2+score3^3)). 
    #' If 4 score columns are provided 3rd and 4th scores are both cubed. 
    #' 
    #' @param data: a dataframe containing columns in score_cols
    #' @param parent_col: a column in data that contains ref ID's.
    #' @param child_cols: a list of columns in data to use for the child cols (orderd by which child should provide the most influence on the final score).
    exp <- (1+data[[score_cols[2]]]^2)
    if(length(score_cols) == 3){
        exp <- (1+data[[score_cols[2]]]^2+data[[score_cols[3]]]^3)
    }
    if(length(score_cols) == 4){
        exp <- (1+data[[score_cols[1]]]^2+data[[score_cols[2]]]^3+data[[score_cols[3]]]^3)
    }
    return(1-((1-data[[score_cols[1]]])^(exp)))
}


calc_combined_scores <- function(scores_data,  combine_score_cols, replace_na_score_cols = NULL, file_out = NULL) {
    score_cols <- unlist(c(combine_score_cols, names(replace_na_score_cols),replace_na_score_cols), use.names = F)
    if(length(replace_na_score_cols) > 0) {
            #replace NA in first column in coalesce_score_cols with values of second column 
            # for every pair of columns in the list
            scores_data <- ungroup(scores_data) %>%
                mutate(!!!map_dfc(replace_na_score_cols,
                               ~replace(.y[[.x[1]]], is.na(.y[[.x[1]]]),
                                        .y[[.x[2]]][is.na(.y[[.x[1]]])]), .y = .))
            
    }
    scores_data <- scores_data %>% ungroup() %>%
        mutate(across(all_of(score_cols) , ~replace_na(.x, 0))) %>%
        mutate(map_dfc(combine_score_cols, ~overall_similarity(.y, .x), .y = .))
    if(length(file_out) > 0) {
        write_csv(scores_data, file_out, na = "")
    }
    return(scores_data)
    
}





