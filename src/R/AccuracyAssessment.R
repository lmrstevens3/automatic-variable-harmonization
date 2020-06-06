#library(plyr)
library(data.table)
library(dplyr)
library(purrr)
library(parallel)
library(multidplyr)
library(xlsx)

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
isCorrectMatch <- function(pairedVariable, possibleCorrectPairings){
    #mark if variable pairing is correct
    grepl(paste0('(^|; )', pairedVariable, '(; |$)'), possibleCorrectPairings)
}

assignCorrectMatches <- function(allScoresData, ref_var_ID, paired_var_ID, correct_pairings, parallel = T, cores = (detectCores() - 2)) {
    cat("starting correct match assignment")
    print(timestamp())
    system.time({
        allScoresData %>%
        {if(parallel = T) {
             #partition for parallel processing and set parallel parameters
            set_default_cluster(makeCluster(cores))
            multidplyr::partition(., allScoresData, !!as.name(ref_var_ID)) 
        } else { 
            dplyr::group_by(., !!as.name(ref_var_ID)) 
        }} %>% 
            group_by(!!as.name(paired_var_ID)) %>% 
            dplyr::mutate(correctMatchTrue = isCorrectMatch(!!as.name(paired_var_ID), !!as.name(correct_pairings))) %>%
            {if(parallel = T) {
                #stop parallel processing
                multidplyr::collect(.)          
                on.exit(stopCluster(cluster))
            } else {.}} %>% 
            dplyr::ungroup()   
    })
    cat("correct match assignment complete")
    return(ungroup(rankData))
    
}








posPredictionsForCut <- function(cutOff, data,  id_col, rank_col) {
    length(data[[id_col]][which(data[[paste0(rank_col)]] <= cutOff)])
}

posPredictionsForAllCut <- function(data, rnk_cuts, id_col, rank_col){
    dplyr::mutate(data, !!!imap(rnk_cuts, function(cutOff, name) name = posPredictionsForCut(cutOff, ...)))
}



calcPositivePredictions <- function(data_descRanked, pairing_rankCutOffs, score_col, ref_var_ID, paired_var_ID, paired_dataset_ID, conceptID) {
    rnk_cuts <- set_names(pairing_rankCutOffs, paste0('top', pairing_rankCutOffs, "predMade_", score_col, sep = ""))
    rank_col <- paste0("rank_", score_col)
    #for data within a group, get the  number of predictions made for each rank cut off in pairing_rankCutOffs vector
    rankData  <-  data_descRanked %>% group_by(!!as.name(conceptID)) %>% 
        group_by(ref_var_ID) %>% 
        split(group_indices(.)) %>% #applies the predMadeInGroupForRankCutOffs function to each group (reference variable (study_1.dataset_1.varID))
        purrr::map_df(posPredictionsForAllCut, rnk_cuts, paired_var_ID, rank_col) 
   return(rankData)
}


getRankData <- function(data, score_col, getPredMade = T, pairing_rankCutOffs = c(1,5,10)){
    #groups each variable1 (variable 1 = one variable in one study  file/dataset) and groups those pairings by studyDataset2 (study2 by possible datasets in that study)
    rankData <- data %>% group_by(concept,  dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% 
        arrange(desc(!!as.name(score_col)), .by_group = T)  %>% 
        mutate(!!as.symbol(paste0("rank_",score_col)) := dense_rank(desc(!!as.name(score_col)))) %>% 
        mutate_at(paste0("rank_",score_col), funs(replace(., !!as.name(score_col) == 0, NA))) %>% 
        group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% 
        mutate(!!as.symbol(paste0("numPairingsRanked_",score_col)) := sum(!is.na(!!as.name(paste0("rank_",score_col)))))
    #get predictions made for each reference variable for different rank cut offs
    if(getPredMade){rankData  <- getPredMadeForRankCutOffs(rankData, pairing_rankCutOffs, score_col)}
    #makes sure rank column name is for the correct score and order columns 
    rankData <- rankData %>% dplyr::select(grep(paste0(score_col,"$"), colnames(.), invert = T), matches(score_col,"$"))
    return(rankData)
}



addRankColsForEachScore <- function(data, score_variables, getPredMade = T, pairing_rankCutOffs = NULL){
    for(i in score_variables){
        print(paste0("calculating ", i, " ranks"))
        data <- getRankData(data, i)
    }
    return(data)
}

# varname <- paste("petal", n , sep=".")
# 
# df <- mutate(iris, !!varname := Petal.Width * n)
# 
# 
# 
 rnk_cuts <- set_names(c(100,125,175), paste0('top', c(100,125,175), "hp", sep = ""))
# 
# avrc <- set_names(c(1,5,10), paste0('top', c(1,5,10), "TP", sep = ""))
# 

mtcars %>% group_by(cyl) %>% 
     split(group_indices(.)) %>% #applies the predMadeInGroupForRankCutOffs function to each group (reference variable (study_1.dataset_1.varID))
     purrr::map_df(predMadeInGroupForRankCutOffs, rnk_cuts, id_col = "mpg",  rank_col = "hp") 
# 
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
# predMadeInGroupForRankCutOffs <- function(.x, rnk_cuts){
#     .x %>% mutate(!!!imap(rnk_cuts, 
#                           function(rnk_cuts, name, grp_data) name = length(grp_data$cyl[grp_data$wt <= rnk_cuts]),
#                     grp_data  = .x)
#     )
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
# iris %>% group_by(Species) %>% 
#     do(invoke_map_dfc(list(map_df),
#                       list(list(select(., c("Species")), unique), 
#                       list(select(., c(matches("(?:Width|Length)"))), sum,  na.rm = T)
#                       )
# ))                                                
                                                     



grdc <- getRankData(rankData, pairing_rankCutOffs = c(1,5,10), score_col = score_col)
oldColNames <- grep("score_desc$", colnames(rankData2), value = T)
names(oldColNames) <- gsub(".predMatches", "predMade", gsub("numRankedMatches", "numPairingsRanked", oldColNames))
rd2 <- rankData2 %>% select(grep(paste0("_score_"), colnames(.), invert = T), oldColNames)#matches(score_col,"$")) %>% rename(oldColNames)
all.equal(rd2, grdc)
grdc %>% filter(dbGaP_studyID_datasetID_varID_1 %in% rd2$dbGaP_studyID_datasetID_varID_1[c(2192058, 2191679, 359659, 359607, 359602, 79416)]) %>% select(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2, varID_2, numPairingsRanked_score_desc)

accuracyByRefVar <- function(accuracyData, scoreVariable){
    accuracyVars <- grep(paste0("_",scoreVariable,"$"), colnames(accuracyData), value =T)
    accuracyData[, gsub(paste0("_",scoreVariable),"", accuracyVars)] <- accuracyData[, accuracyVars]
    #create True Positive boolean variables for vars with correct match and <= to rank cut off for each cut off in rank_cuts
    TPboolVars <- set_names(paste0('top', pairing_rankCutOffs, "TP_1varPair"), paste0('top', pairing_rankCutOffs, "TP_1var1dataset"))
    accuracyData <- accuracyData %>% ungroup() %>% 
        mutate(!!!imap(TPboolVars, function(TPboolVars, name, data) 
            name = sum(unique(TP_bool_vars), na.rm = T)/length(data$dbGaP_studyID_datasetID_varID_2),
            data  = .)
        )
    accuracyTotalsByVar <- accuracyData %>% group_by(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% 
        #calculates true positive on dataset level (if a dataset has multiple possible correct variable pairings, if any correct pairing is in top ranks, TP =  1, and does not count multiple correc pariings for one dataset))
        mutate(!!!imap(set_names(paste0('top', pairing_rankCutOffs, "TP_1varPair"), paste0('top', pairing_rankCutOffs, "TP_1var1dataset"), 
                                 function(TPboolVars, name, data) 
                                     name = sum(unique(TP_bool_vars), na.rm = T)/length(data$dbGaP_studyID_datasetID_varID_2),
                                 data  = .))
               ) %>% group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% #get accuracy data by variable 
        do(invoke_map_dfc(list(map_df), 
                          list(list(select(., c(starts_with("total"), "numPairingsRanked",  matches("top\\d+predMade"))), unique), 
                               list(select(., c(matches("top\\d+(?:TP_1varPair|TP_1var1dataset)"))), sum,  na.rm = T))
        )) %>% ungroup()
    return(accuracyTotalsByVar)
}



precision <- function(data, truePositives, predictedPositives){
    precision = data[[truePositives]]/data[[predictedPositives]]
    return(precision)
}

recall <- function(data, truePositives, actualPositives){
    recall = data[[truePositives]]/data[[actualPositives]]
    return(recall)
}

F1 <- function(data, truePositives, predictedPositives, actualPositives){
    precision <- precision(data, truePositives, predictedPositives)
    recall <- precision(data, truePositives, actualPositives)
    F1 <- (2*precision*recall)/(precision+recall)
    F1stats <- tibble("precision" = precision, "recall" = recall, "F1" = F1)
    return(F1stats)
    
}

accuracyByConcept <- function(accuracyTotalsByVarData, F1parameters) {
    accuracyTotalsByConcept <- accuracyTotalsByVarData %>% group_by(concept) %>% 
        #get total var/dataset, num pairings, parings ranked, TP, and predictions made counts by concept
        do(invoke_map_dfc(list(map_df), 
                          list(list(select(., c(starts_with("total"))), unique), 
                               list(select(., c("numPairingsRanked", matches("top\\d+(?:TP_1varPair|TP_1var1dataset|predMade)"))), sum,  na.rm = T))
        )) %>% split(group_indices(.)) %>% 
        #calculate precision, recall, and F1
        purrr::map_df(.f = function(.x) {
            .x %>% mutate(!!!map(F1parameters, 
                                 function(F1parameters, name, grp_data) name = list(do.call("F1", args = F1parameters)),
                                 grp_data = .x, name = names(F1parameters))) 
                          }
            ) %>% tidyr::unnest(.sep="_") %>%
        mutate_at(vars(matches("_F1")), funs(replace(.,is.na(.),0))) %>% ungroup()
}


accuracyTotals <- function(accuracyData, scoreVariable, F1parameters){
    #get accuracy by reference variable
    accuracyTotalsByVarData <- accuracyByRefVar(accuracyData, scoreVariable)
    #get accuracy by concept
    accuracyTotalsByConcept <- accuracyByConcept(accuracyTotalsByVarData, F1parameters)
    #get total accuracy
    conceptCols <- grep("InConcept", colnames(accuracyTotalsByConcept), value = T)
    conceptColsRename <- setNames(conceptCols, gsub("InConcept", "", conceptCols))
    accuracyTotals <- accuracyTotalsByConcept %>%
        dplyr::summarize_at(c(starts_with("total"), "numPairingsRanked", matches("top\\d+(?:TP_1varPair|TP_1var1dataset|predMade)")), sum) %>%
        #calculate overall precision, recall, and F1
        mutate(!!!map(F1parameters, 
                      function(F1parameters, name, grp_data) name = list(do.call("F1", args = F1parameters)),
                      grp_data = .x, name = names(F1parameters))
               ) %>% tidyr::unnest(.sep="_") %>%
        mutate_at(vars(matches("_F1")), funs(replace(.,is.na(.),0))) %>% 
        mutate(concept = "All Concepts") %>% 
        rename(!!conceptColsRename)
    accuracyResults <- accuracyTotalsByConcept %>% bind_rows(accuracyTotals)
    return(accuracyResults)
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
scoresFile <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/FHS_CHS_MESA_ARIC_all_similarity_scores_and_combined_scores_ManuallyMappedConceptVars_Mar2020.csv'
manualConceptVarMapFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv'
rankFileOut <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/FHS_CHS_MESA_ARIC_all_scores_ranked_ManuallyMappedConceptVars_June2020.csv'
accuracyXlsxFileOut = '/Users/laurastevens/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/accuracy_VarMappingsBagOfWordsAllScores_June2020.xlsx'


#scores data file 
#!!note!!- important to make sure na.strings and quotes are read in the same for files to merge correctly- files are written with NA values as "" for interoperability, and some strings have double and single quote characters
allScoresData <- fread(scoresFile , header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))
#mannually mapped concepts and possible matches
conceptMappedVarsData <- fread(manualConceptVarMapFile, header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))
tbl_df(conceptMappedVarsData) %>% dplyr::select(conceptID, study_1, dbGaP_dataset_label_1, varID_1, var_desc_1, totalVarsInConcept, totalDatasetsInConcept, ends_with("Dataset"), ends_with("Matches"))

#Join scores data and concept data and filter out datasets not in goldstandard and concepts with broad definitions that don't map across variables/studies (ex. cholesterol lowering med because not same medication for mappings)
#takes about 3-4 min with 21267777 X 41 and 1703 X 21
allScoresData <- allScoresData %>% dplyr::left_join((conceptMappedVarsData %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1, -var_coding_labels_1))) %>% 
    filter(!concept %in% c("Cholesterol Lowering Medication")) %>% group_by(concept) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1)

#"unique Participant Identifier" (could also remove if desired)
#check join worked correctly
unique(allScoresData$concept) #50 concepts, 0.99 GB, 2367346 X 49 after merge and filter (#51 concepts, 9218935 X 49, 4.2 GB if use PID concept) (#concepts Time  To CVDDeath and Time To CHDDeath filtered out because they are only present in mesa)
object.size(allScoresData)

#create correct match variable for accuracy and 
timestamp() #started 4:35pm
rankData <- prepScoresDataForAccuracy(allScoresData) 
timestamp()
rm(allScoresData) #remove all scores data so its not taking up memory

cores <- detectCores() - 2 # macbook pro can split 4 cores into 8 cores for parallel procesing
myfunc  <- function(data) {
    dplyr::mutate(reference_variable = paste0(unique(na.omit(c(study_1, dbGaP_dataset_label_1, var_desc_1))), collapse = "."), #create ref/match var for display/graphing later
                  paired_variable = paste0(unique(na.omit(c(study_2, dbGaP_dataset_label_2, var_desc_2))), collapse = ".")) %>%
        dplyr::select(data, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2, 
                     correctMatchTrue, conceptID,  reference_variable, paired_variable, concept, conceptID,
                     study_1, study_2, dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_2,
                     varID_1, varID_2, units_1, units_2, var_coding_counts_distribution_1, var_coding_counts_distribution_2, 
                     timeIntervalDbGaP_1, timeIntervalDbGaP_2,  metadataID_1, metadataID_2, matchID, 
                     numConceptVarsInDataset, pairingsInSameDataset, totalVarsInConcept, totalDatasetsInConcept, 
                     totalPossiblePairingsInConcept, totalPossiblePairingsInConcept_1var1dataset, starts_with("score_"))
}
filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1) 
#get ranks for all scores
possibleScores <- grep("^score_",colnames(allScoresData)[-c(which(colnames(allScoresData)  %in% c("score_euclidianDistance")))], value = T)
timestamp()#takes about 25 min with input of 9218935 X 45 (16 scores total)
rankData  <- addRankColsForEachScore(allScoresData, possibleScores, getPredMade = T, pairing_rankCutOffs = c(1,5,10)) 
timestamp()
fwrite(rankData, rankFileOut, sep = ',', qmethod = 'double', na = "", row.names = F)


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
F1parameters <- c(f1vars_1varPair, f1vars_1var1dataset)

timestamp()
accuracyTotals <- sapply(possibleScores, function(x){print(paste0("calculating ", x, " accuracy")); accuracyTotals(rankData2, x)}, USE.NAMES = T, simplify = F)
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

