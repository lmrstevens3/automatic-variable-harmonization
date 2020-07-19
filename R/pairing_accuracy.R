#library(plyr)
library(data.table)
library(dplyr)
library(purrr)
library(tidyr)
library(parallel)
library(multidplyr)
library(xlsx)
library(docs)
library(DT)
library(plotly)
library(processx)

# c("CHD", "AF", "race_ethnicity",  "age", "diabetes", "SBP", "DBP")
# rankData %>% filter(conceptID %in% c("CHD"), rank_score_desc == 1) %>% dplyr::select(concept, dbGaP_studyID_datasetID_varID_1, varID_1, dbGaP_studyID_datasetID_2, varID_2, correctMatchTrue, rank_score_desc, top1.predMatches_score_desc) 
# t2 <- t %>% group_by(concept) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1) %>% mutate(pred = length(dbGaP_studyID_datasetID_varID_2))
# t %>% group_by(concept) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1) %>% group_by(dbGaP_studyID_datasetID_varID_1) %>% mutate(pred = length(dbGaP_studyID_datasetID_varID_2[which(rank_score_desc == 1)]))
# #234 matches for CHD concept including datasets not in gold standard
# t <-rankData %>% filter(conceptID == "CHD", rank_score_desc == 1) #%>% dplyr::select(concept, dbGaP_studyID_datasetID_varID_1, varID_1, dbGaP_studyID_datasetID_2, varID_2, correctMatchTrue, rank_score_desc, topPredMatches_score_desc, dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_varID_2) 
# chkRanksMade <- rankData2 %>% select(concept, dbGaP_studyID_datasetID_1, varID_1, dbGaP_studyID_datasetID_2, varID_2, rank_score_desc, rank_score_units, rank_score_desc_SqNoisyOr_maxOtherMeta_euclidean, dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_varID_2, dbGaP_studyID_datasetID_varID_1)
# #getting how top.predMatches is caclucated
# t2 <-t %>% group_by(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% mutate(numVarsInMatchedDataset = length(dbGaP_studyID_datasetID_2)) %>%
#     group_by(dbGaP_studyID_datasetID_varID_1) %>% mutate(top1.pred  = sum(topPredMatches_score_desc/numVarsInMatchedDataset)) %>% select(-rank_score_desc)
# 
# rankData %>% group_by(concept) %>% filter(conceptID == "CHD", rank_score_desc == 1, dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1) %>% 
#     mutate(datasetsInConcept = length(unique(dbGaP_studyID_datasetID_2))) %>% 
#     dplyr::select(conceptID, reference_variable, matched_variable,  ends_with("Dataset"), ends_with("Matches"), correctMatchTrue, rank_score_desc, datasetsInConcept) 

#check combined sores are correct- check codelab >0 and  score_euc  == 0, check noisy or for something 
#calculate toppredmatches_1var_dataset by #5,10,1*(numVarsInConcept)(numD-1)
#use t and check topPredmatches by variable for rankData
#then check race/age/stroke/marital_status
#then run all

is_correct_pair <- function(paired_ID_col, correctPairings, sep = "; "){
    #' Helper function to determine if variable id is in a string of id's separated by semi_colon. 
    #' Can be used in assignedCorrectPairings instead of %in% command in mutate. 
    #' 
    #' @param pairedVariable: the ID of suggested variable pairing.
    #' @param correctPairings: the string of IDs that are correct pairings separated by sep.
    grepl(paste0('(^|',sep,')', paired_ID_col, '(',sep,'|$)'), correctPairings)
}


assign_correct_pairings <- function(data, ref_ID_col, paired_ID_col, class_col =  NULL, parallel = T, cores = (detectCores() - 2)) {
    #' Evaluates if a pairing is correct in a dataframe of pairings and appends a boolean column called 'correct_pairing' to the data frame. 
    #' Option to parallelize for large dataframes. Uses a group_var to define which set of paired_ID's should be considered correct. 
    #' The function isCorrect above can be called for correct_match, a list of paired_IDs are provided in form of a concatenated string (requires  rowwise grouping and is much slower).
    #' 
    #' @param data: a dataframe containing a column of ref ID and a column of corrrect pairings (paired_ID) to that ref ID.
    #' @param paired_ID_col: the column in pariningData that contains pairing IDs.
    #' @param ref_ID_col: the column in pariningData that contains ref IDs.
    #' @param class_col: the grouping variable that defines which set of paired IDs should be compared to the ref ID.
    #' 
    #' @returns: data with appended boolean column called 'correct_pairing'.
    print("starting correct match assignment")
    timestamp()
    system.time({
        data <- data %>% 
        {if (parallel) {
            #partition for parallel processing and set parallel parameters
            set_default_cluster(makeCluster(cores))
            multidplyr::partition(., !!as.name(class_col))
        } else {.}} 
        %>% group_by_at(., vars(class_col)) %>% 
            mutate(correct_pairing = !!as.name(paired_ID_col) %in% !!as.name(ref_ID_col)) %>%
        {if(parallel) {
            multidplyr::collect(.)
            on.exit(stopCluster(cluster)) #stop parallel processing
        } else {.}} 
    })
    cat("correct match assignment complete")
    return(ungroup(data))
}


n_unique <- function(x){
    return(length(unique(x)))
}

clac_n_pairings <- function(x, y = NULL){
    if(length(y) == 0){y = x}
    return((n_unique(x)*(n_unique(y)- 1)))
}

calc_true_pairings_n <- function(data, ref_ID_col, class_col = NULL, ref_multiple_group_col = NULL, multiple_correction = F) {
    #' Calculates the total number of actual pairings for the ref IDs in ref_ID_col and does not count ref ID paired with itself.
    #' If a calss is provided, only ref ID pairings within a class are counted and not pairings between classes. 
    #' If a pairable col is provided, ref ID with the same value in pairable_col are not counted. 
    #' 
    #' @param data: a dataframe containing a column of ref ID to pair together. 
    #' @param ref_ID_col: a column in data that contains ref ID's.
    #' @param class_col: a grouping variable that defines sets of ref IDs should be counted when paired.
    #' @param multiple_group_col: a grouping variable that defines ref_ID belonging to the same group. 
    #' If supplied pairs of ref ID with the same multiple_group_col value are not counted.
    #' @param multiple_correction: logical value indicating if multiple_correction should be made for multiple ref ID in the same multiple_group_col group. 
    #' When TRUE, pairings from ref ID x, paired with ref IDs y1:yn, where y1:yn all have the same multiple_group_col value (not equal to ref ID x multiple_group_col value) are counted as a single pairing.
    #' 
    #' @returns: data with appended columns: unique ref ID count (<ref_ID_col>_n) and total actual pairings (true_pairings_n). 
    #' If ref_multiple_group_col is supplied, unique ref_multiple_group_col count (<ref_ID_col>_n_ref_multiple_group_col) is also appended
    #' If class_col is supplied, data contains counts by class 
    try(if(length(ref_multiple_group_col) == 0 & multiple_correction) stop("multiple_group_col required if multiple_correction = T"))
    if(!length(ref_multiple_group_col) == 0) {ref_in_mult  <- paste0(paste0(class_col, ref_ID_col, collapse = "_"), '_in_', ref_multiple_group_col)}
    data <-  data %>% group_by_at(class_col) %>%
        mutate_at(c(ref_ID_col, ref_multiple_group_col), funs("n" = n_unique(.))) %>%
        {if(!length(ref_multiple_group_col) == 0) {
            group_by_at(c(class_col, ref_multiple_group_col)) %>% 
                mutate(!!as.symbol(ref_in_mult) := n_unique(ref_ID_col)) %>% group_by_at(class_col) %>%
            {if(multiple_correction) {
                mutate(true_pairings_n = clac_n_pairings(!!as.name(ref_ID_col),!!as.name(ref_multiple_group_col)))
            } else {
                mutate(true_pairings_n = clac_n_pairings(!!as.name(ref_ID_col)) - sum(!!as.name(ref_in_mult)-1))
            }}
        } else {
            mutate(true_pairings_n = clac_n_pairings(!!as.name(ref_ID_col)))               
        }} %>% ungroup () %>%
    return(data)
}


calc_pos_predict_for_cut <- function(cutoff, data, rank_col, paired_ID_col) {
    length(data[[paired_ID_col]][which(data[[rank_col]] <= cutoff)])
}

calc_correct_pair_for_cut <- function(cutoff, data, rank_col, correct_pairing) {
    !(is.na(data[[rank_col]]) && data[[rank_col]] >= cutoff && (!data[[correct_pairing]]))
}

calc_pos_predict_all_cut <- function(data, cutoffs, rank_col, paired_ID_col, correct_pairing) {
    cutoffs_tp <- setNames(cutoffs, gsub('_pred_', '_true_', names(cutoffs)))
    dplyr::mutate(data, !!!purrr::imap(cutoffs, function(cutoff, name) name = calc_pos_predict_for_cut(cutoff, data, rank_col, paired_ID_col))) %>% 
    dplyr::mutate(!!!purrr::imap(cutoffs_tp, function(cutoff, name) name = calc_correct_pair_for_cut(cutoff, data, rank_col, correct_pairing)))
}


calc_positive_predictions <- function(data, rank_col, ref_ID_col, paired_ID_col, correct_pairings_col, pairing_rank_cutoffs = c(1,5,10), paired_multiple_group_col = NULL, multiple_correction = F) {
    #' calculates the number positive predictions made and true positives for each reference ID (number of pairings with a similarity score > 0). 
    #' 
    #' @param data: data frame of pairings with a column of ranks. 
    #' @param rank_col: the column in data that contains pairing ranks.
    #' @param ref_ID_col: the column in data that contains ref IDs.
    #' @param paired_ID_col: the column in data that contains pairing IDs.
    #' @param pairing_rankCutOffs: a vector of rank cut off values when calculating positive predictions and true positives
    #' @param paired_multiple_group_col: a grouping variable that defines paired_IDs belonging to the same group.
    #' @param multiple_correction: logical value indicating if multiple_correction should be made for multiple ref ID in the same multiple_group_col group. 
    #' When TRUE, multiple true positive pairings from ref ID x, paired with ref IDs y1:yn, where y1:yn all have the same multiple_group_col value are counted as one true positive.
    #' 
    #' @returns: data appended with columns topX_pred_pos, topX_true_pos for each x in pairing_rank_cutoffs.
    
    try(if(length(paired_multiple_group_col) == 0 & multiple_correction) stop("multiple_group_col required if multiple_correction = T"))
    cutoffs <- setNames(pairing_rank_cutoffs, paste0('top', pairing_rank_cutoffs, "_pred_pos", sep = ""))
    
    #for data within a group, get the  number of predictions and true positives for each rank cut off in pairing_rankCutOffs vector
    data  <-  data %>% group_by(!!as.name(ref_ID_col)) %>% 
        split(group_indices(.)) %>% #applies the calcPosPredictAllCut function to each group (ref_ID)
        purrr::map_df(calc_pos_predict_all_cut, cutoffs, rank_col, paired_ID_col, correct_pairings_col) %>%
        {if(multiple_correction) {
            group_by(., !!as.name(paired_multiple_group_col), add = T) %>%
            mutate_at(matches("_true_pos"), sum(unique(.), na.rm = T)/length(!!as.name(paired_ID_col)))
        } else {mutate_at(matches("_true_pos"), sum(unique(.), na.rm = T))}} %>% 
        dplyr::select(c(ref_ID_col, class_col, ref_multiple_group_col, matches("_n$|_pred_pos|true_pos"))) %>% 
        unique()
   return(data)
}




# varname <- paste("petal", n , sep=".")
# 
# df <- mutate(iris, !!varname := Petal.Width * n)
# 
# 
# 
# cuts <- setNames(c(100), paste0('top', c(100), "hp"))
# cuts2 <- setNames(c(50), paste0('top', c(50), "hp"))
# 
# avrc <- set_names(c(1,5,10), paste0('top', c(1,5,10), "TP", sep = ""))
# 

# mtcars %>% group_by(cyl) %>%
#      split(group_indices(.)) %>% #applies the predMadeInGroupForRankCutOffs function to each group (reference variable (study_1.dataset_1.varID))
#      purrr::map_df(calc_pos_predict_all_cut, cuts, rank_col = "wt", paired_ID_col = "cyl", correct_pairing = "mpg")
# # 
# applyForAllRankCutOffs <- function(.x, namedVarParameterList, ){}
# 
# 
# mtcars %>% group_by(cyl, gear) %>% 
#     do(invoke_map_dfc(list(map_df), 
#                       list(list(select(., c("hp", "carb", "vs",  "am")), length), 
#                            list(select(., c(ends_with("t"), ends_with("p"))), sum))
#     ) 
#     )
# 
# 
# testCuts <- function(.x, cuts){
#     cuts2 <- cuts
#     names(cuts2) <- "this_is_new_name"
#     .x %>% mutate(!!!purrr::imap(cuts,
#                           function(cuts, name, grp_data) name = length(grp_data$cyl[grp_data$wt <= cuts]),
#                     grp_data  = .x)) %>% mutate(!!!purrr::imap(cuts2,
#                                                               function(cuts2, name) name = .x$mpg*cuts2))
# 
# }

    




# pr.t <- sapply(c("Sepal","Petal"), function(x){
#     PRvars <- as.list(c(as.name("grp_data"), 
#                         paste0(x, c(".Width", ".Length", ".Length"))))
#     names(PRvars) <- c("data", "truePositives", "predictedPositives", "actualPositives")
#     return(PRvars)
# }, simplify = F, USE.NAMES = T)
# pr.t2 <- sapply(c("Sepal","Petal"), function(x){
#     PRvars <- as.list(c(as.name("grp_data"), 
#                         paste0(x, c(".Width", ".Length", ".Width"))))
#     names(PRvars) <- c("data", "truePositives", "predictedPositives", "actualPositives")
#     return(PRvars)
# }, simplify = F, USE.NAMES = T)
# names(pr.t2) <- c("Sepal2","Petal2")
# prt <- c(pr.t, pr.t2)
# iris %>% group_by(Species) %>%
#     split(group_indices(.)) %>%
#     purrr::map_df(.f = function(.x, name) {
#         .x %>% mutate(!!!map(prt, function(prt, name, grp_data) {
#                                             name = list(do.call("F1", args = prt))
#                                     }, grp_data = .x, name = names(prt)))
#     }, name = names(pr.t)) %>% tidyr::unnest(.sep="")
#  
# iris %>%
#     do(purrr::invoke_map_dfc(list(purrr::map_df),
#                       list(#list(select(., c("Species")), unique),
#                       list(select(., c(matches("(?:Width|Length)"))), sum,  na.rm = T)
#                       )
# ))
#                                                      
# 
# 
# 
# grdc <- getRankData(rankData, pairing_rankCutOffs = c(1,5,10), score_col = score_col)
# oldColNames <- grep("score_desc$", colnames(rankData2), value = T)
# names(oldColNames) <- gsub(".predMatches", "predMade", gsub("numRankedMatches", "numPairingsRanked", oldColNames))
# rd2 <- rankData2 %>% select(grep(paste0("_score_"), colnames(.), invert = T), oldColNames)#matches(score_col,"$")) %>% rename(oldColNames)
# all.equal(rd2, grdc)
# grdc %>% filter(dbGaP_studyID_datasetID_varID_1 %in% rd2$dbGaP_studyID_datasetID_varID_1[c(2192058, 2191679, 359659, 359607, 359602, 79416)]) %>% select(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2, varID_2, numPairingsRanked_score_desc)
# 


calc_precision <- function(data, truePositives, predictedPositives){
    precision = data[[truePositives]]/data[[predictedPositives]]
    return(precision)
}

calc_recall <- function(data, truePositives, actualPositives){
    recall = data[[truePositives]]/data[[actualPositives]]
    return(recall)
}

calc_F1 <- function(data, truePositives, predictedPositives, actualPositives){
    precision <- precision(data, truePositives, predictedPositives)
    recall <- precision(data, truePositives, actualPositives)
    F1 <- (2*precision*recall)/(precision+recall)
    F1stats <- tibble("precision" = precision, "recall" = recall, "F1" = F1)
    return(F1stats)
}

calc_F1_precision_recall_all <- function(F1params, name, data){
    dplyr::mutate(data, !!!map(F1params, function(F1params, name) name = list(do.call("calc_F1", args = F1params))))
} 

calc_accuracy <- function(data, rank_col, ref_ID_col, paired_ID_col, correct_pairings_col, pairing_rank_cutoffs = c(1,5,10), paired_multiple_group_col = NULL, multiple_correction = F, class_col = NULL, ref_multiple_group_col = NULL) {
    #' calculates positive predicitions, true positives, accuracy (precision, recall, F1) for each class in data. 
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param rank_col: column name of similarity score ranks in data
    #' @param paired_ID_col: the column data that contains pairing IDs
    #' @param ref_ID_col: the column in data that contains ref ID's
    #' @param correct_pairings_col: logical column in data indicating if a pairing is correct or not
    #' @param pairing_rankCutOffs a vector of rank cut off values when calculating positive predictions.
    #' @param class_col: a grouping variable that defines sets of pairable ref IDs when calculating total acutal pairings and accuracy
    #' @param ref_multiple_group_col: a grouping variable that defines ref_IDs belonging to the same group. 
    #' @param paired_multiple_group_col: a grouping variable that defines paired_IDs belonging to the same group. 
    #' @param multiple_correction: logical value indicating if multiple_correction should be made for multiple ref ID in the same ref_multiple_group_col group, and 
    #' multiple paired_ID in the same group paired to a ref ID. When TRUE, pairings from ref ID x, paired with paired IDs y1:yn, where y1:yn 
    #' all have the same paired_multiple_group_col value are counted as a single pairing and if any pairings in the set of x,y1:yn are correct, it is counted as one true positive.
    #' 
    #' @returns: a dataframe with the stats listed below. If class col provided, stats are also calculated by class
    #' with first row of the dataframe is overall accuracy for all classes
    #' dataframe includes counts (n) for: 
    #' refID and ref_multiple_group_col/class_col (if provided), 
    #' true pairings, true positives, and positive predictions 
    #' parings ranked (only included if column ending in "ranked_n" is included in data passed in- this column is calculated if rank_scores() in rank_pairing_scores file is used to rank data), 
    #' F1, precision, recall 
    
    #calculate number of true positives, positive predictions, and true_pairings
    data <- calc_positive_predictions(data, rank_col, ref_ID_col, paired_ID_col, correct_pairings_col, pairing_rank_cutoffs, paired_multiple_group_col, multiple_correction)
    data <- calc_true_pairings_n(data, ref_ID_col, class_col, ref_multiple_group_col, multiple_correction) %>% 
        dplyr::select(-c(paste0(ref_ID_col, '_n', ref_multiple_group_col)))
    unique_cols <- c(paste0(c(ref_ID_col, ref_multiple_group_col), "_n"), "true_pairings_n")
    #match true positives, positive predictions, and true_pairings to F1 parameters for all rank cut offs
    params_F1 <- sapply(paste0('top', c(1,5,10)), function(x){
        setNames(as.list(c(as.name("data"), c(paste0(x, c("_true_pos", "_pred_pos")), "true_pairings_n"))), 
                 c("data", "truePositives", "predictedPositives", "actualPositives"))
    }, simplify = F, USE.NAMES = T)
    
    #calculate accuracy stats
    data <- data %>% group_by_at(!!as.name(class_col)) %>% 
        #get counts (n) (by class if class_col provided): refID/ref_multiple_group_col (if provided), true pairings, parings ranked, true positives, and positive predictions 
        do(invoke_map_dfc(list(map_df), 
                          list(list(select(., c(unique_cols, unique))), 
                               list(select(., c(matches("_ranked_n|_true_pos|_pred_pos)"))), sum,  na.rm = T))
        )) %>% split(group_indices(.)) %>% 
        #calculate precision, recall, and F1
        purrr::map_df(calc_F1_precision_recall_all(params_F1, name, .x)) %>% tidyr::unnest(.sep="_") %>%
        ungroup() %>%
        #add  overall accuracy if accuracy was calculated by class
        {if(!length(class_col) == 0) {
            bind_rows(. %>% dplyr::summarize_at(vars(matches(paste0(class_col, "|F1|precision|recall"))), sum) %>%
                          purrr::map_df(calc_F1_precision_recall_all(params_F1, name, .)) %>% tidyr::unnest(.sep="_") %>%
                          mutate(concept = paste("All Concepts ", "N = ", length(unique(.[[class_col]])))))                    
        } else {.}} %>% mutate_at(vars(matches("_F1")), funs(replace(.,is.na(.), 0)))
    return(data)
}



####OLD FUNCTIONS
accuracyByRefVar1 <- function(accuracyData, scoreVariable){
    rankVars <- grep(paste0("_",scoreVariable,"$"), colnames(accuracyData), value =T)
    accuracyData[, gsub(paste0("_",scoreVariable),"",rankVars)] <- accuracyData[,rankVars]
    #determine if top 1,5,10 ranked vars has correct match 
    accuracyData <- accuracyData %>% ungroup() %>% dplyr::mutate(top10 = ifelse(correctMatchTrue == T & (!is.na(rank)) & rank <= 10, T, F), top5 = ifelse(correctMatchTrue == T & (!is.na(rank)) & rank <= 5, T, F), top1 = ifelse(correctMatchTrue == T & (!is.na(rank)) & rank == 1, T, F))
    #get var accuracy
    accuracyTotalsByVar <- accuracyData %>% group_by(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% 
        mutate(top10.TP_1var1dataset = sum(unique(top10), na.rm = T)/length(dbGaP_studyID_datasetID_varID_2), top5.TP_1var1dataset = sum(unique(top5), na.rm = T)/length(dbGaP_studyID_datasetID_varID_2), top1.TP_1var1dataset = sum(unique(top1), na.rm = T)/length(dbGaP_studyID_datasetID_varID_2)) %>%
        group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% 
        dplyr::summarize(dbGaP_studyID_datasetID_1 = unique(dbGaP_studyID_datasetID_1),
                         totalPossibleMatches = unique(totalPossibleMatches), totalPossibleMatches_1var1dataset = unique(totalPossibleMatches_1var1dataset), numPairingsRanked = unique(numPairingsRanked), 
                         totalVarsInConcept = unique(totalVarsInConcept), totalDatasetsInConcept = unique(totalDatasetsInConcept),
                         top10.TP = sum(top10, na.rm = T), top10.TP_1var1dataset = sum(top10.TP_1var1dataset), top10.pred = unique(top10predMade),
                         top5.TP = sum(top5, na.rm = T), top5.TP_1var1dataset = sum(top5.TP_1var1dataset), top5.pred = unique(top5predMade), 
                         top1.TP = sum(top1, na.rm = T), top1.TP_1var1dataset = sum(top1.TP_1var1dataset), top1.pred  = unique(top1predMade))
    return(accuracyTotalsByVar)
}

accuracyTotals1 <- function(accuracyData, scoreVariable){
    #get accuracy by reference variable
    accuracyTotalsByVar <- refVarAccuracy(accuracyData, scoreVariable)
    #get accuracy by concept
    accuracyTotalsByConcept <- accuracyTotalsByVar %>% group_by(concept) %>% 
        dplyr::summarize(totalVarsMatched = unique(totalVarsInConcept), totalDatasets = unique(totalDatasetsInConcept), numPairingsRanked = sum(numPairingsRanked),
                         totalPossibleMatches = unique(totalPossibleMatches),  totalPossibleMatches_1var1dataset = unique(totalPossibleMatches_1var1dataset),
                         top10.TP = sum(top10.TP, na.rm = T), top10.TP_1var1dataset = sum(top10.TP_1var1dataset), top10.predictMade = sum(top10.pred), top10.predictMade_1var1dataset = totalPossibleMatches_1var1dataset, 
                         top10.precision = top10.TP/top10.predictMade, top10.recall = top10.TP/totalPossibleMatches, top10.precision.recall_1var1dataset = top10.TP_1var1dataset/totalPossibleMatches_1var1dataset, 
                         top10.F1 = 2*top10.precision*top10.recall/(top10.precision+top10.recall), top10.F1_1var1dataset = 2*top10.precision.recall_1var1dataset*top10.precision.recall_1var1dataset/(top10.precision.recall_1var1dataset+top10.precision.recall_1var1dataset),
                         top5.TP = sum(top5.TP, na.rm = T), top5.TP_1var1dataset = sum(top5.TP_1var1dataset), top5.predictMade = sum(top5.pred), top5.predictMade_1var1dataset = totalPossibleMatches_1var1dataset,
                         top5.precision = top5.TP/top5.predictMade, top5.recall = top5.TP/totalPossibleMatches, top5.precision.recall_1var1dataset = top5.TP_1var1dataset/totalPossibleMatches_1var1dataset, 
                         top5.F1 = 2*top5.precision*top5.recall/(top5.precision+top5.recall), top5.F1_1var1dataset = 2*top5.precision.recall_1var1dataset*top5.precision.recall_1var1dataset/(top5.precision.recall_1var1dataset+top5.precision.recall_1var1dataset),
                         top1.TP = sum(top1.TP, na.rm = T), top1.TP_1var1dataset = sum(top1.TP_1var1dataset), top1.predictMade  = sum(top1.pred), top1.predictMade_1var1dataset = totalPossibleMatches_1var1dataset,
                         top1.precision = top1.TP/top1.predictMade, top1.recall = top1.TP/totalPossibleMatches, top1.precision.recall_1var1dataset = top1.TP_1var1dataset/totalPossibleMatches_1var1dataset, 
                         top1.F1 = 2*top1.precision*top1.recall/(top1.precision+top1.recall), top1.F1_1var1dataset = 2*top1.precision.recall_1var1dataset*top1.precision.recall_1var1dataset/(top1.precision.recall_1var1dataset+top1.precision.recall_1var1dataset)) %>%
        mutate_at(vars(matches(".F1")), funs(replace(.,is.na(.),0)))
    #get total accuracy
    accuracyTotals <- accuracyTotalsByConcept  %>% 
        dplyr::summarize(totalVarsMatched = sum(totalVarsMatched), totalDatasets = sum(totalDatasets), numPairingsRanked = sum(numPairingsRanked), totalPossibleMatches = sum(totalPossibleMatches), totalPossibleMatches_1var1dataset = sum(totalPossibleMatches_1var1dataset), 
                         top10.TP = sum(top10.TP, na.rm = T), top10.TP_1var1dataset = sum(top10.TP_1var1dataset), top10.predictMade = sum(top10.predictMade), top10.predictMade_1var1dataset  = sum(top10.predictMade_1var1dataset), top10.precision = top10.TP/top10.predictMade, top10.recall = top10.TP/totalPossibleMatches, top10.precision.recall_1var1dataset = top10.TP_1var1dataset/totalPossibleMatches_1var1dataset, top10.F1 = 2*top10.precision*top10.recall/(top10.precision+top10.recall), top10.F1_1var1dataset = 2*top10.precision.recall_1var1dataset*top10.precision.recall_1var1dataset/(top10.precision.recall_1var1dataset+top10.precision.recall_1var1dataset),
                         top5.TP = sum(top5.TP, na.rm = T), top5.TP_1var1dataset = sum(top5.TP_1var1dataset), top5.predictMade = sum(top5.predictMade), top5.predictMade_1var1dataset  = sum(top5.predictMade_1var1dataset),  top5.precision = top5.TP/top5.predictMade, top5.recall = top5.TP/totalPossibleMatches, top5.precision.recall_1var1dataset = top5.TP_1var1dataset/totalPossibleMatches_1var1dataset, top5.F1 = 2*top5.precision*top5.recall/(top5.precision+top5.recall), top5.F1_1var1dataset = 2*top5.precision.recall_1var1dataset*top5.precision.recall_1var1dataset/(top5.precision.recall_1var1dataset+top5.precision.recall_1var1dataset),
                         top1.TP = sum(top1.TP, na.rm = T), top1.TP_1var1dataset = sum(top1.TP_1var1dataset), top1.predictMade  = sum(top1.predictMade), top1.predictMade_1var1dataset  = sum(top1.predictMade_1var1dataset), top1.precision = top1.TP/top1.predictMade, top1.recall = top1.TP/totalPossibleMatches, top1.precision.recall_1var1dataset = top1.TP_1var1dataset/totalPossibleMatches_1var1dataset, top1.F1 = 2*top1.precision*top1.recall/(top1.precision+top1.recall), top1.F1_1var1dataset = 2*top1.precision.recall_1var1dataset*top1.precision.recall_1var1dataset/(top1.precision.recall_1var1dataset+top1.precision.recall_1var1dataset)) %>% 
        mutate(concept = "All Concepts") %>% mutate_at(vars(matches(".F1")), funs(replace(.,is.na(.),0)))
    accuracyResults <- accuracyTotalsByConcept %>% bind_rows(accuracyTotals)
    return(accuracyResults)
}


#get accuracy differences
differences <- function(data, variablesToCompare){
    combos <- as.data.frame(combn(variablesToCompare,2))[1:length(variablesToCompare)-1]
    print(combos)
    for(i in 1:length(combos)){
        var1 <- as.character(combos[1,i])
        var2 <- as.character(combos[2,i])
        data <- eval(substitute(mutate(data, diff = var2 - var1), 
                                    list(var1 = as.name(var1), var2 = as.name(var2))))
        colnames(data)[which(colnames(data) %in% "diff")] <- paste0(var1,".DIFF_FROM.", var2)
    }
    return(data)
}


#################################################################################
################ Accuracy Main Script
manualConceptVarMapFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv'
rankFile <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_scores_ranked_manually_mapped_vars.csv'
accuracyXlsxFileOut = '/Users/laurastevens/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/accuracy_VarMappingsBagOfWordsAllScores_June2020.xlsx'


rankData <- fread(rankFile , header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))

#scores data file 
#!!note!!- important to make sure na.strings and quotes are read in the same for files to merge correctly- files are written with NA values as "" for interoperability, and some strings have double and single quote characters
allScoresRankData <- fread(scoresFile , header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))
#mannually mapped concepts and possible matches
conceptMappedVarsData <- fread(manualConceptVarMapFile, header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))
tbl_df(conceptMappedVarsData) %>% dplyr::select(conceptID, study_1, dbGaP_dataset_label_1, varID_1, var_desc_1, totalVarsInConcept, totalDatasetsInConcept, ends_with("Dataset"), ends_with("Matches"))

#Join scores data and concept data and filter out datasets not in goldstandard and concepts with broad definitions that don't map across variables/studies (ex. cholesterol lowering med because not same medication for mappings)
#takes about 3-4 min with 21267777 X 41 and 1703 X 21
allScoresRankData  <- allScoresRankData  %>% dplyr::left_join((conceptMappedVarsData %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1, -var_coding_labels_1))) %>% 
    filter(!concept %in% c("Cholesterol Lowering Medication")) %>% group_by(concept) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1)

#"unique Participant Identifier" (could also remove if desired)
#check join worked correctly
unique(allScoresRankData $concept) #50 concepts, 0.99 GB, 2367346 X 49 after merge and filter (#51 concepts, 9218935 X 49, 4.2 GB if use PID concept) (#concepts Time To CVDDeath and Time To CHDDeath filtered out because they are only present in mesa)
object.size(allScoresRankData )

#create correct match variable for accuracy and 
rankData <- assign_correct_pairings(rankData, ref_ID_col = "dbGaP_studyID_datasetID_varID_1", paired_ID_col = "dbGaP_studyID_datasetID_varID_2", class_col =  "concept", parallel = F, cores = NULL)
accuracyData <- 


accuracyTotals <- sapply(score_cols, function(x) {
    print(paste0("calculating ", x, " accuracy")) 
    calc_accuracy(rankData, rank_col = paste0("rank_", x), ref_ID_col = "dbGaP_studyID_datasetID_varID_1", paired_ID_col = "dbGaP_studyID_datasetID_varID_2", correct_pairings_col = "correct_pairing", pairing_rank_cutoffs = c(1,5,10), paired_multiple_group_col = "dbGaP_studyID_datasetID_2", multiple_correction = T, class_col = "concept", ref_multiple_group_col = "dbGaP_studyID_datasetID_1") 
    }, USE.NAMES = T, simplify = F)


#rm(allScoresData) #remove all scores data so its not taking up memory

cores <- detectCores() - 2 # macbook pro can split 4 cores into 8 cores for parallel procesing



#get precision, recall, and F1-started 1:25pm
#accuracyData <- rankData2 %>% group_by(dbGaP_studyID_datasetID_varID_1) %>% dplyr::filter(correctMatchTrue == T)
#define variables and F1 parameters for each 1var1pair and 1var1dataset case and each top rank cut off
f1vars_1varPair <- sapply(paste0('top', c(1,5,10), '_1varPair'), function(x){
    f1vars <- as.list(c(as.name("grp_data"), 
                        paste0(gsub('_1varPair',"",x), c("TP_1varPair", "predictMade", "totalPossiblePairingsInConcept"))))
    names(f1vars) <- c("data", "truePositives", "predictedPositives", "actualPositives")
    return(f1vars)
}, simplify = F, USE.NAMES = T)
f1vars_1var1dataset <- sapply(paste0('top', c(1,5,10), '_1var1dataset'), function(x){
    f1vars <- as.list(c(as.name("grp_data"), 
                        paste0(gsub('_1var1dataset',"",x), c("TP_1var1dataset", "totalPossiblePairingsInConcept_1var1dataset", "totalPossiblePairingsInConcept_1var1dataset"))))
    names(f1vars) <- c("data", "truePositives", "predictedPositives", "actualPositives")
    return(f1vars)
}, simplify = F, USE.NAMES = T)
TPboolVars <- set_names(paste0('top', pairing_rankCutOffs, "TP_1varPair"), 
                        paste0('top', pairing_rankCutOffs, "TP_1var1dataset"))
F1parameters <- c(f1vars_1varPair, f1vars_1var1dataset)
c(starts_with("total"), "numPairingsRanked",  matches("top\\d+posPred")))
timestamp()
accuracyTotals <- sapply(possibleScores, function(x){print(paste0("calculating ", x, " accuracy")); accuracyTotals(rankData, x)}, USE.NAMES = T, simplify = F)
accuracyTotals[["overallAccuracyAllScores"]] <- do.call(rbind, accuracyTotals) %>% filter(concept == "All Concepts") %>% dplyr::mutate(score = names(accuracyTotals)) %>% 
    dplyr::select(score, totalVarsMatched, totalPossibleMatches, totalPossibleMatches_1var1dataset, numPairingsRanked, ends_with(".TP"), ends_with(".TP_1var1dataset"), ends_with(".precision"), ends_with(".recall"), ends_with("recall_1var1dataset"), ends_with("F1"), ends_with("F1_1var1dataset"))
timestamp()
#write results to a file
write.xlsx(as.data.frame(accuracyTotals[[length(accuracyTotals)]]), accuracyXlsxFileOut, sheetName = names(accuracyTotals[length(accuracyTotals)]), row.names = F)
sapply(names(accuracyTotals)[-length(accuracyTotals)], function(accuracyName){
    write.xlsx2(x=as.data.frame(accuracyTotals[[accuracyName]]), file=accuracyXlsxFileOut, sheetName = gsub("score_","",accuracyName), row.names = F, append = T)
})

#get differences in accuracy by concept and plot 
accuracyDifferencesData <- lapply(accuracyTotals[c(1,11,13,15)], function(x){returnData <- as.data.frame(x %>% dplyr::select(top1.precision.recall_1var1dataset)); rownames(returnData) <- x$concept; return(returnData)})
accuracyDifferencesData <- do.call('cbind', accuracyDifferencesData) 
accuracyDifferences.top1recall <- differences(accuracyDifferencesData %>% select(ends_with(".recall")), grep(".recall", colnames(accuracyDifferencesData), value = T))
rownames(accuracyDifferences.top1recall) <- rownames(accuracyDifferencesData)

plot_ly(x = accuracyDifferences.top1recall$score_desc.top1.recall.DIFF_FROM.score_desc_SqNoisyOr_maxOtherMeta_euclidean.top1.recall*(100), y = row.names(accuracyDifferences.top1recall), type = 'bar') %>% 
    layout(xaxis = list(title = "Recall Difference"), title ="variables scored by descriptions vs. combined with max score from variables scored on units/codelabels/distribtuion", font = list(size = 8))
plot_ly(x = accuracyDifferences.top1recall$score_desc.top1.recall.DIFF_FROM.score_descUnits_SqNoisyOr_codeLabEuclidean.top1.recall*(100), y = row.names(accuracyDifferences.top1recall), type = 'bar') %>% 
    layout(xaxis = list(title = "Recall Difference"), title ="variables scored by descriptions vs. descriptions+units scores combined with codelabels/distribtuion scores", font = list(size = 8))
plot_ly(x = accuracyDifferences.top1recall$score_desc.top1.recall.DIFF_FROM.score_desc_SqNoisyOr_codeLabEuclidean.top1.recall*(100), y = row.names(accuracyDifferences.top1recall), type = 'bar') %>% 
    layout(xaxis = list(title = "Recall Difference"), title ="variables scored by descriptions vs. descriptions scores combined with codelabels/distribtuion scores", font = list(size = 8))

