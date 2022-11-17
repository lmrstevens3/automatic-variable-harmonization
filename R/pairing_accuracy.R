library(tidyverse)
library(parallel)
library(multidplyr)


is_correct_pair <- function(paired_ID_col, correctPairings, sep = "; "){
    #' Helper function to determine if variable id is in a string of id's separated by semi_colon. 
    #' Can be used in assignedCorrectPairings instead of %in% command in mutate. 
    #' 
    #' @param pairedVariable: the ID of suggested variable pairing.
    #' @param correctPairings: the string of IDs that are correct pairings separated by sep.
    grepl(paste0('(^|',sep,')', paired_ID_col, '(',sep,'|$)'), correctPairings)
}

assign_true_pairing <- function(data, ref_ID_col, paired_ID_col, class_cols =  NULL, parallel = F, cores = (parallel::detectCores() - 2)) {
    #' Evaluates if a pairing is correct in a dataframe of pairings and appends a boolean column called 'correct_pairing' to the data frame. 
    #' Option to parallelize for large dataframes. Uses a group_var to define which set of paired_ID's should be considered correct. 
    #' @param data: a dataframe containing a column of ref ID and a column of corrrect pairings (paired_ID) to that ref ID.
    #' @param paired_ID_col: the column in pariningData that contains pairing IDs.
    #' @param ref_ID_col: the column in pariningData that contains ref IDs.
    #' @param class_col: the grouping variable(s) that defines which set of paired IDs should be compared to the ref ID.
    #' 
    #' @returns: data with appended boolean column called 'correct_pairing'.
    data %>% dplyr::group_by(., across(all_of(class_cols))) %>% 
        {if (parallel) {
            #partition for parallel processing and set parallel parameters
            multidplyr::partition(., new_cluster(cores))
        } else {.}} %>%
            dplyr::mutate(., true_pairing = (!!as.name(paired_ID_col)) %in% (!!as.name(ref_ID_col))) %>%
        {if(parallel) collect(.)
        else .} %>% 
        ungroup()
    #on.exit(closeAllConnections()) 
}


n_unique <- function(x){
    #' Calculate the number of unique values of x (size of the set)
    return(length(unique(x)))
}

calc_n_pairings <- function(x, y = NULL){
    #' Calculates the possible combinations of x not including x paired with itself (x*x-1).
    #' If y provided calculates the possible combinations of x and y, assuming x can not pair with the y it is associated with (x*y-1).
    if(length(y) == 0){y = x}
    return((n_unique(x)*(n_unique(y)- 1)))
}

summarize_actual_pairings <- function(data, ref_ID_col, class_col = NULL, ref_group_col = NULL, multiple_correction = F) {
    #' Calculates the total number of actual true pairings for the ref IDs in ref_ID_col and does not count ref ID paired with itself.
    #' If a calss is provided, only ref ID pairings within a class are counted and not pairings between classes. 
    #' If ref_group_col is provided ref ID's within same ref group are not counted, and 
    #'  when multiple correction is set to true, groups with multiple refID's are normalized 
    #'  such that parings with sets of ref ID's belonging to the same ref group equate to one.
    #' 
    #' @param data: a dataframe containing a column of ref ID to pair together. 
    #' @param ref_ID_col: a column in data that contains ref ID's.
    #' @param class_col: a grouping variable that defines sets of ref IDs should be counted when paired.
    #' @param ref_group_col: a grouping variable that defines ref_ID belonging to the same group. 
    #' If supplied, pairs of ref ID with the same ref group are not counted.
    #' @param multiple_correction: logical value indicating if correction should be made for multiple ref ID's in the same ref group. 
    #' When TRUE, pairings ref ID x, paired with ref IDs y1:yn in ref group y are counted once.
    #' 
    #' @returns: data with appended columns: unique ref ID count (<ref_ID_col>_n) and total actual pairings (true_pairings_n). 
    #' If ref_group_col is supplied, unique ref_multiple_group_col count (<ref_ID_col>_n_ref_group_col) is also appended
    #' If class_col is supplied, data contains counts by class 
    if(multiple_correction & length(ref_group_col) == 0) stop("ref_group_col required if multiple_correction = T")
    if(!length(ref_group_col) == 0) {n_ref_in_refgroup  <- paste0(c("n", class_col, ref_ID_col, 'in_group'), collapse = "_")}
    data %>% group_by(across(any_of(class_col))) %>%
        dplyr::mutate(across(any_of(c(ref_ID_col, ref_group_col)), ~n_unique(.x), .names = "n_{.col}")) %>%
        {if(!length(ref_group_col) == 0) {
            dplyr::group_by(., across(all_of(c(class_col, ref_group_col)))) %>% 
                dplyr::mutate(!!as.symbol(n_ref_in_refgroup) := n_unique(!!as.name(ref_ID_col))) %>% 
                dplyr::group_by(across(any_of(class_col))) %>%
                dplyr::mutate(across(all_of(n_ref_in_refgroup), ~sum(.x[!duplicated(!!as.name(ref_ID_col))] - 1), .names = "n_same_group_pairs")) %>%
            {if(multiple_correction) {
                dplyr::mutate(., actual_pairings = !!as.name(paste0("n_", ref_group_col)) - 1,
                          n_actual_pairings = calc_n_pairings(!!as.name(ref_ID_col),!!as.name(ref_group_col)))
            } else {
                dplyr::mutate(., actual_pairings = !!as.name(paste0("n_", ref_ID_col)) - !!as.name(n_ref_in_refgroup), 
                          n_actual_pairings = calc_n_pairings(!!as.name(ref_ID_col)) - n_same_group_pairs)
            }}
        } else {
            dplyr::mutate(., actual_pairings = !!as.name(paste0("n_", ref_ID_col)) - 1, 
                      n_actual_pairings = calc_n_pairings(!!as.name(ref_ID_col)))               
        }} %>% 
        dplyr::ungroup() %>% unique()
}




calc_true_positives <- function(actual_true, predicted_true) {
    #' calculates boolean true positives
    actual_true & predicted_true
}


summarize_predictions <- function(data, ref_cols, paired_ID_col, true_pairing_col, rank_col, rank_cutoffs = c(1,5,10), multiple_correction = F, paired_group_col = NULL, class_col = NULL) {
    #' calculates the number predictions made and true positives for each reference ID and each class if class_col is provided
    #' 
    #' @param data: data frame of pairings with a column of ranks. 
    #' @param rank_col: the column in data that contains pairing ranks.
    #' @param ref_ID_col: the column(s) in data containing ref IDs (columns w ref information that should be perserved, such as ref groups).
    #' @param paired_ID_col: the column in data that contains pairing IDs.
    #' @param rank_cutoffs: a vector of rank cut off values when calculating positive predictions and true positives
    #' @param multiple_correction: logical value indicating if correction should be made for multiple ref ID's in the same ref group. 
    #' When TRUE, pairings ref ID x, paired with ref IDs y1:yn in ref group y are counted once.
    #' @param paired_group_col: a grouping variable that defines paired_IDs belonging to the same group.
    #' @param class_col: a grouping variable that defines sets of ref IDs should be counted when paired.
    #' @returns: data appended with columns topX_pred_pos, topX_true_pos for each x in pairing_rank_cutoffs.
    
    if(multiple_correction & length(paired_group_col) == 0) stop("paired_group_col required if multiple_correction = T")
    if(!rank_col %in% colnames(data)) stop("rank_col must be a column in data")
    if(!true_pairing_col %in% colnames(data)) stop("true_pairing_col must be a column in data")
     
    pred_cuts <- setNames(rank_cutoffs, paste0('top', rank_cutoffs, "_pred", sep = ""))
    true_pred_cols <- set_names(names(pred_cuts), gsub("pred", "true_pred", names(pred_cuts)))

    data %>% dplyr::group_by(across(all_of(ref_cols))) %>%
        #calc boolean values for valid predictions based on rank cutoff and true predictions
        dplyr::mutate(across(all_of(rank_col), ~sum(!is.na(.x)), .names = "{.col}_pairs")) %>%
        dplyr::filter(if_all(all_of(rank_col), ~(.x <= max(rank_cutoffs) | sum(is.na(.x)) == length(.x)))) %>% 
        dplyr::ungroup() %>% 
        dplyr::mutate(map_dfc(pred_cuts, ~((!is.na(.y)) & .y <= .x), .y = .[[rank_col]])) %>% 
        dplyr::mutate(map_dfc(select(., all_of(true_pred_cols)), 
                       ~calc_true_positives(.y, .x), .y = .[[true_pairing_col]])) %>% 
        #calculate total predictions and true predictions for each refID
        dplyr::group_by(across(all_of(ref_cols))) %>%
        {if(multiple_correction) {
            dplyr::mutate(., across(matches("top\\d+_(?:pred|true)"), ~length(unique((!!as.name(paired_group_col))[.x]))))
        } else { 
            dplyr::mutate(., across(matches("top\\d+_(?:pred|true)"), ~sum(.x, na.rm = T)))
        }} %>% 
        dplyr::select(all_of(c(class_col,ref_cols, paste0(rank_col, "_pairs"))),  matches("_pred$")) %>% 
        unique() %>% 
        #get total counts (n) (by class_col if provided):
        dplyr::group_by(across(any_of(class_col))) %>% 
        dplyr::mutate(across(matches("n_|_pred$"), sum, na.rm = T, .names = "n_{.col}")) %>% 
        dplyr::ungroup()
}




calc_precision <- function(true_positives, predicted_positives){
    replace_na(true_positives/predicted_positives, 0)
}

calc_recall <- function(true_positives, actual_positives){
    replace_na(true_positives/actual_positives, 0)
}

calc_F1 <- function(true_positives, predicted_positives, actual_positives, name = NULL){
    #' calculates precision, recall, F1
    precision <- calc_precision(true_positives, predicted_positives)
    recall <- calc_recall(true_positives, actual_positives)
    F1 <- replace_na((2*precision*recall)/(precision+recall), 0)
    set_names(list(precision, recall, F1), 
              paste0(name, c("precision", "recall", "F1")))
}



calc_accuracy <- function(data, ref_ID_col, paired_ID_col, true_pairing_col, rank_col, rank_cutoffs = c(1,5,10), class_col = NULL, ref_group_col = NULL, multiple_correction = F, paired_group_col = NULL, remove_refID_stats = T) {
    #' calculates positive predicitions, true positives, accuracy (precision, recall, F1) for each class in data. 
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param rank_col: column name of similarity score ranks in data
    #' @param paired_ID_col: the column data that contains pairing IDs
    #' @param ref_ID_col: the column in data that contains ref ID's
    #' @param true_pairing_col: logical column in data indicating if a pairing is correct or not
    #' @param pairing_rankCutOffs a vector of rank cut off values when calculating positive predictions.
    #' @param class_col: a grouping variable that defines sets of pairable ref IDs when calculating total acutal pairings, predictions, and accuracy stats
    #' @param ref_group_col: a grouping variable that defines ref_IDs belonging to the same group. 
    #' @param paired_group_col: a grouping variable that defines paired_IDs belonging to the same group. 
    #' @param multiple_correction: logical value indicating if multiple_correction should be made for multiple ref ID in the same ref_group_col group, and 
    #' multiple paired_ID in the same group paired to a ref ID. When TRUE, pairings from ref ID x, paired with paired IDs y1:yn, where y1:yn 
    #' all have the same paired_group_col value are counted as a single pairing and if any pairings in the set of x,y1:yn are correct, it is counted as one true positive.
    #' @param include_stats_by_ref_ID: If TRUE accuracy stats by individual ref ID are provided including total stats and stats by class
    #' 
    #' @returns: a dataframe with the stats listed below. If class col provided, stats are also calculated by class
    #' with first row of the dataframe is overall accuracy for all classes. If include_stats_by_refID is True, additional stats by refID are included. 
    #' dataframe includes counts (n) for: 
    #' refID and ref_group_col-if provided, 
    #' actual true, true prediction, and predicitons
    #' pairings ranked,
    #' F1, precision, recall 
    
    message(timestamp(prefix = "  ##---", suffix = ": calculating actual and prediction stats---##", quiet = T))
    #calculate number of true positives, positive predictions, and true_pairings
    data <- summarize_predictions(data, c(ref_ID_col, ref_group_col), paired_ID_col, true_pairing_col, rank_col, rank_cutoffs, multiple_correction, paired_group_col, class_col) %>%
        summarize_actual_pairings(ref_ID_col, class_col, ref_group_col, multiple_correction) %>% 
        {if(remove_refID_stats) {
            dplyr::select(., any_of(class_col), matches("^n_"), -matches("n_.*in_group|same_group_pairs$")) 
        } else {.}} %>%
        dplyr::ungroup() %>% unique() %>% 
        dplyr::arrange(across(any_of(c(class_col, ref_id_col))))
    
    #set F1 cols 
    pred_cols <- grep("top\\d+_pred", colnames(data), value = T)
    true_pred_cols <- grep("top\\d+_true_pred", colnames(data), value = T)
    actual_cols <- sort(rep(grep("actual_pairings", colnames(data), value = T), length(rank_cutoffs)))
    
    #sum refID/class totals if class provided
    totals <- {if(is.null(class_col)) NULL 
               else {data %>% 
                       dplyr::summarise(across(matches("^n_"), ~sum(.x[!duplicated(!!as.name(class_col))], na.rm = T)),
                                across(matches("^top|^actual|_pairs$"), ~sum(.x, na.rm = T)),
                                across(all_of(class_col), ~paste0("All ", class_col, " (N = ", n_unique(.x), ")")))
              }}
            
    message(timestamp(prefix = "  ##---", suffix = ": calculating F1 stats---##", quiet = T)) 
    #calculate accuracy (F1) stats
    dplyr::bind_rows(totals, data) %>%
        dplyr::mutate(., pmap_dfc(list(true_pred_cols, pred_cols, actual_cols),
                           ~calc_F1(..4[[..1]], ..4[[..2]], ..4[[..3]], gsub("^n_|true.*$", "", ..1)), ..4 = .)) %>% 
        dplyr::select(any_of(c(class_col, ref_ID_col)), everything()) 
    
}

combine_accuracy_results <- function(all_accuracy, totals_only = F, totals_col = "concept", totals_regex = "All concept") {
    lapply(1:length(all_accuracy), function(x) {
        all_accuracy[[x]] %>% 
            select(concept, matches("_(?:precision|recall|F1)")) %>%
            mutate(score_type = gsub("_|score|rank", "", names(all_accuracy)[x]))
    }) %>% bind_rows(.) %>% 
        {if(totals_only) {
            filter(.,if_all(all_of(totals_col), ~grepl(totals_regex, .x)))
        } else {.}}
}

#get accuracy differences
calc_cols_differences <- function(data, cols_to_compare){
    combos <- as.data.frame(combn(cols_to_compare,2))[1:length(cols_to_compare)-1]
    for(i in 1:length(combos)){
        var1 <- as.character(combos[1,i])
        var2 <- as.character(combos[2,i])
        data <- eval(substitute(mutate(data, diff = var2 - var1), 
                                    list(var1 = as.name(var1), var2 = as.name(var2))))
        colnames(data)[which(colnames(data) %in% "diff")] <- paste0("diff_", var2,"_", var1)
    }
    return(data)
}



