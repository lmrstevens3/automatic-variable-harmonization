scoresFile <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/FHS_CHS_MESA_ARIC_all_similarity_scores_and_combined_scores_ManuallyMappedConceptVars_Mar2020.csv'
rankFileOut <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/FHS_CHS_MESA_ARIC_all_scores_ranked_ManuallyMappedConceptVars_June2020.csv'

{if (parallel = T) {
    #partition for parallel processing and set parallel parameters
    set_default_cluster(makeCluster(cores))
    multidplyr::partition(., !!as.name(class_col))
} else {
    group_by(!!as.name(class_col))
}} %>% mutate(correct_match = !!as.name(paired_ID_col) %in% !!as.name(ref_ID_col)) %>%
{if(parallel = T) {
    multidplyr::collect(.)
    on.exit(stopCluster(cluster)) #stop parallel processing
} else {.}} 


rank_scores <- function(data, score_col, ref_ID_col, rank_by_col = NULL){
    #' sorts data by score_col (descending) and dense ranks pairings with highest similarity scores. 
    #' Pairings with a score of zero recieve a rank value of NA. A rank_<score_col> column is appended to data 
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param score_col: the column name of the column in data containing similarity scores
    #' @param ref_ID_col: the column in data that contains unique ref IDs
    #' @rank_by_col: A grouping variable that defines if ranking 1:n should be done by group instead of for all pairings for a given ref ID. 
    #' For example, if scores for a ref come from multiple sources, the column listing the scores source could be supplied, 
    #' and ref ID would be ranked 1:n scores for each source instead of 1:n scores in all sources
    rankData <- data %>% group_by(!!as.name(ref_ID_col), !!as.name(rank_by_col)) %>% 
        arrange(desc(!!as.name(score_col)), .by_group = T)  %>% 
        mutate(!!as.symbol(paste0("rank_",score_col)) := dense_rank(desc(!!as.name(score_col)))) %>% 
        mutate_at(paste0("rank_",score_col), funs(replace(., !!as.name(score_col) == 0, NA))) %>% 
        group_by(ref_ID_col) %>% 
        mutate(!!as.symbol(paste0(score_col, "_ranked_n")) := sum(!is.na(!!as.name(paste0("rank_",score_col)))))
    
    rankData <- rankData %>% dplyr::select(grep(paste0(score_col,"$"), colnames(.), invert = T), matches(score_col,"$"))
    return(rankData)
}

addRankColsForEachScore <- function(data, score_variables, ref_ID_col, pairable_col = NULL){
    #wrapper function to call calculate Ranks and append a set of rank cols for multiple pairing scores
    for(i in score_variables) {
        print(paste0("calculating ", i, " ranks"))
        data <- rank_scores(data, i, ...)
    }
    return(data)
}

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
#groups each variable1 (variable 1 = one variable in one study  file/dataset) and groups those pairings by studyDataset2 (study2 by possible datasets in that study)
rankData  <- addRankColsForEachScore(allScoresData, possibleScores, getPredMade = T, pairing_rankCutOffs = c(1,5,10)) 
timestamp()
fwrite(rankData, rankFileOut, sep = ',', qmethod = 'double', na = "", row.names = F)