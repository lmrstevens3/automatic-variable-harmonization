library(dplyr)
scoresFile <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/FHS_CHS_MESA_ARIC_all_similarity_scores_and_combined_scores_ManuallyMappedConceptVars_Mar2020.csv'
rankFileOut <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/FHS_CHS_MESA_ARIC_all_scores_ranked_ManuallyMappedConceptVars_June2020.csv'
manualConceptVarMapFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv'



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
mannual_map_ref_data <- fread(manualConceptVarMapFile, header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))

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


test_concepts <- c(
    'AF', 
    'Age (years)', 
    'alcohol intake (gm)', 
    'CHD', 
    'CVD', 
    'Death', 
    'Education', 
    'Gender', 
    'Race/Ethnicity', 
    'Time To AF', 
    'Total Caloric Intake (kCal)'
)
test_scores_rank_data <-  rankData %>% filter(concept %in% test_concepts) %>%
    select(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_1, 
           dbGaP_studyID_datasetID_varID_2, dbGaP_studyID_datasetID_2, correctMatchTrue, 
           score_desc,  rank_score_desc, score_codeLab_relativeDist, rank_score_codeLab_relativeDist) %>% 
    unique()


test_mannual_map_ref_data <-  mannual_map_ref_data %>% filter(concept %in% test_concepts) %>%
    select(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_1, varDocID_1, study_1, 
           units_1, var_desc_1, var_coding_counts_distribution_1, possiblePairingsInConcept)

write.csv(test_mannual_map_ref_data, "Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/src/tests/test_mannual_map_ref_data.csv", row.names = F)
write.csv(test_scores_rank_data, "Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/src/R/tests/test_var_similarity_scores_rank_data.csv", row.names = F)


test_accuracy_data <-  test_scores_rank_data %>% 
        select(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_1, 
           dbGaP_studyID_datasetID_varID_2, dbGaP_studyID_datasetID_2, rank_score_desc, correctMatchTrue) %>%
        group_by(concept) %>%
        mutate(ref_ID_n = length(unique(dbGaP_studyID_datasetID_varID_1)), 
               ref_dataset_n = length(unique(dbGaP_studyID_datasetID_1))) %>% 
        group_by(concept, dbGaP_studyID_datasetID_1) %>% 
        mutate(concept_ref_ID_in_dataset = length(unique(dbGaP_studyID_datasetID_varID_1))) %>% 
        filter(rank_score_desc <= 5) %>%
        group_by(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>%
        mutate(top1_true_pos = sum(unique(correctMatchTrue[which(rank_score_desc <= 1)]), na.rm = T)/length(unique(dbGaP_studyID_datasetID_varID_2)),
               top5_true_pos = sum(unique(correctMatchTrue[which(rank_score_desc <= 5)]), na.rm = T)/length(unique(dbGaP_studyID_datasetID_varID_2))) %>%
        group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% 
        mutate(top1_true_pos =  sum(top1_true_pos), 
               top5_true_pos =  sum(top5_true_pos),
               top1_true_pos2 =  sum(correctMatchTrue[which(rank_score_desc <= 1)], na.rm = T), 
               top5_true_pos2 =  sum(correctMatchTrue[which(rank_score_desc <= 5)], na.rm = T),
               top1_pred_pos2 =  sum(length(dbGaP_studyID_datasetID_varID_2[which(rank_score_desc <= 1)]), na.rm = T), 
               top5_pred_pos2 =  sum(length(dbGaP_studyID_datasetID_varID_2[which(rank_score_desc <= 5)]), na.rm = T)) %>%
        group_by(concept) %>% 
        mutate(pairs_in_concept_n = n(), 
               same_dataset_pairs = sum(concept_ref_ID_in_dataset[!duplicated(dbGaP_studyID_datasetID_varID_1)]-1)) %>%
        select(-concept_ref_ID_in_dataset)

        
            
        
test_accuracy_data %>% filter(concept == test_concepts[6])  %>% select("same_dataset_pairs", matches("concept|_n$|_dataset_n$|_pos2$"), dbGaP_studyID_datasetID_varID_1) %>% unique()  %>% mutate_at(vars(matches("_pos$|_pos2$")), sum)                        
test_accuracy_data %>% filter(concept == test_concepts[1])  %>% select("same_dataset_pairs", matches("concept|_n$|_dataset_n$|_pos$"), dbGaP_studyID_datasetID_varID_1) %>% unique()  %>% mutate_at(vars(matches("_pos$|_pos2$")), sum)                        

test_scores_rank_data %>% filter(concept == test_concepts[1])

#test if any paired_IDs in ref ID have a correctMatchTrue of False
rankData %>% filter(concept == test_concepts[1], (!correctMatchTrue), 
                    dbGaP_studyID_datasetID_varID_1 %in% conceptMappedVarsData[conceptMappedVarsData$concept == test_concepts[1]]$dbGaP_studyID_datasetID_varID_1 & 
                        dbGaP_studyID_datasetID_varID_2 %in% conceptMappedVarsData[conceptMappedVarsData$concept == test_concepts[1]]$dbGaP_studyID_datasetID_varID_1) %>% 
    select(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2, var_coding_counts_distribution_1, var_coding_counts_distribution_2)
rankData %>% group_by(concept) %>%filter((!correctMatchTrue), dbGaP_studyID_datasetID_varID_2 %in% dbGaP_studyID_datasetID_varID_1)

#check if pairing order changes score
system.time({
    combo_match <- rankData %>% group_by(concept) %>% filter(dbGaP_studyID_datasetID_varID_2 %in% dbGaP_studyID_datasetID_varID_1) %>% 
        select("ref" = dbGaP_studyID_datasetID_varID_1 , "pair" = dbGaP_studyID_datasetID_varID_2, concept, matches("^score_")) %>%
        rowwise() %>% 
        mutate(pairing = paste0(sort(c(ref,pair)), collapse="~"), pairing_order = paste(c("ref", "pair")[match(c(ref,pair),sort(c(ref,pair)))], collapse="~")) %>%
        ungroup()
})

head(combo_match)
table(combo_match$pairing_order)
dim(combo_match %>% filter(pairing_order == "ref~pair"))
dim(combo_match %>% filter(pairing_order == "pair~ref"))
all_equal(combo_match %>% filter(pairing_order == "ref~pair") %>% select(pairing,matches("scores")), 
      combo_match %>% filter(pairing_order == "pair~ref") %>% select(pairing,matches("scores")))

#check when concatenating is helpful
system.time({
scores_info <- rankData %>% filter(correctMatchTrue) %>% 
    mutate_at(c("rank_score_descCodeLabUnits", "rank_score_descUnits", "rank_score_descCodeLab"), funs("better" = . < rank_score_desc)) %>% 
    mutate(euc_better = rank_score_codeLab_euclidean < rank_score_codeLab_relativeDist, 
           noisy_euc_better = rank_score_desc_SqNoisyOr_codeLabEuclidean < rank_score_desc_SqNoisyOr_codeLabRelativeDistance, 
           max_better_rel = rank_score_desc_SqNoisyOr_codeLabEuclidean < rank_score_desc_SqNoisyOr_maxOtherMeta_euclidean, 
           max_better_euc = rank_score_desc_SqNoisyOr_codeLabRelativeDistance < rank_score_desc_SqNoisyOr_maxOtherMeta_relDist) %>% 
    mutate_at(c("score_codeLab_relativeDist", "score_codeLab_euclidean", "score_desc", "score_units"), funs("0_1" = replace(., !(. == 0 | . == 1), "(0,1)" ))) %>%
    mutate(ref_var_continuous = grepl('(?:max|min|mean|median|sd)[ ]*=[ ]*(?:\\d|.\\d)*;+', var_coding_counts_distribution_1), 
           paired_var_continuous = grepl('(?:max|min|mean|median|sd)[ ]*=[ ]*(?:\\d|.\\d)*;+', var_coding_counts_distribution_2)) %>%
    mutate(pairing_data_type = rowSums(select(.,"ref_var_continuous", "paired_var_continuous")),
           pairing_data_type = plyr::mapvalues(pairing_data_type, c(0,1,2), c("categorical", "categorical_continous", "continous"))) %>%
    rowwise() %>%
    mutate(doc_na = paste0(c(c("units1", "code1")[c(is.na(units_1), grepl('(nulls[ ]*=\\d+[; ]*$|[ ]*(?:max|min|mean|median|sd)[ ]*=[ -]*(?:\\d|.\\d)*;+)', var_coding_counts_distribution_1))],
                           c("units2", "code2")[c(is.na(units_2), grepl('(nulls[ ]*=\\d+[; ]*$|[ ]*(?:max|min|mean|median|sd)[ ]*=[ -]*(?:\\d|.\\d)*;+)', var_coding_counts_distribution_2))]), "", collapse = "_"), 
           score_0 = paste0(c(c("desc", "euc", "rel", "units")[c(is.na(rank_score_desc), is.na(rank_score_codeLab_euclidean), is.na(rank_score_codeLab_relativeDist), is.na(rank_score_units))]),"", collapse = "_"),
           score_1 = paste0(c(c("desc", "euc", "rel", "units")[c(score_desc == 1, score_codeLab_euclidean == 1, score_codeLab_relativeDist == 1, score_units == 1)]),"", collapse = "_"))
})
#look at concat  
table(scores_info %>% select(c("rank_score_descUnits_better", "rank_score_descCodeLab_better", "rank_score_descCodeLabUnits_better")))
table(scores_info$rank_score_descCodeLabUnits_better)
table(scores_info$rank_score_descUnits_better, useNA = "ifany")
table(scores_info$rank_score_descCodeLab_better)
table(scores_info$rank_score_descCodeLabUnits_better, scores_info$rank_score_descUnits_better, useNA = "ifany")
table(scores_info$rank_score_descCodeLabUnits_better, scores_info$rank_score_descCodeLab_better, useNA = "ifany")
table(scores_info[scores_info$doc_na == "", "rank_score_descCodeLab_better"], useNA = "ifany")
table(scores_info[scores_info$doc_na == "", "rank_score_descCodeLabUnits_better"], useNA = "ifany")



#look at units present
table(is.na((rankData %>% select(units_1, units_2))))
table(is.na((scores_info %>% select(units_1, units_2))))
  
#look at euc vs. rel 
table(scores_info %>% filter(!rank_score_codeLab_euclidean == rank_score_codeLab_relativeDist) %>% select(euc_better), useNA = "ifany")
table(scores_info %>% filter(!rank_score_codeLab_euclidean == rank_score_codeLab_relativeDist) %>% 
          mutate(euc_1 = score_codeLab_euclidean == 1) %>% 
          select(euc_1), useNA = "ifany")
#euc range is much less granular than rel_dist range when ranks are not equal (3% of pairs) (similar otherwise)
summary(scores_info %>% filter(!rank_score_codeLab_euclidean == rank_score_codeLab_relativeDist) %>% 
            select( var_coding_counts_distribution_1, var_coding_counts_distribution_2, score_codeLab_euclidean, score_codeLab_relativeDist,dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2))
#look at euc vs. rel 
scores_info %>% filter((!euc_better) & (!rank_score_codeLab_euclidean == rank_score_codeLab_relativeDist) & 
                           conceptID %in% c("alcoholIntake", "avgSBP", "heartRate", "coffee"))  %>% 
    group_by(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% 
    mutate_at(c('rank_score_codeLab_euclidean', 'rank_score_codeLab_relativeDist'), funs("max" = max(., na.rm = T))) %>% 
    ungroup() %>% mutate(dist_rank_diff =  rank_score_codeLab_euclidean - rank_score_codeLab_relativeDist) %>% 
    select(dist_rank_diff, concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2)

#look at continous vs categorical 
table(scores_info$score_relativeDistance > 0, scores_info$score_euclidianDistanceScaled > 0, scores_info$score_codeLab > 0)

#look at units distribution and times when units between 0,1 actually help

#look at max vs code/dist
table(scores_info %>% filter(!rank_score_desc_SqNoisyOr_maxOtherMeta_relDist == rank_score_desc_SqNoisyOr_codeLabRelativeDistance) %>% select(max_better_rel), useNA = "ifany")
table(scores_info %>% filter(!rank_score_desc_SqNoisyOr_maxOtherMeta_euclidean == rank_score_descUnits_SqNoisyOr_codeLabEuclidean) %>% select(max_better_euc), useNA = "ifany")


#look at num of time 0,1 dist/codelab or desc scores cause combined score to lower rank
table(scores_info %>% filter(!rank_score_desc_SqNoisyOr_codeLabRelativeDistance == rank_score_descUnits_SqNoisyOr_codeLabEuclidean) %>% select(noisy_euc_better), useNA = "ifany")
table(scores_info %>% filter(!rank_score_desc_SqNoisyOr_codeLabRelativeDistance == rank_score_descUnits_SqNoisyOr_codeLabEuclidean) %>% select(noisy_euc_better, rank_score_codeLab_relativeDist_0_1, rank_score_codeLab_euclidean_0_1), useNA = "ifany") 
summary(scores_info %>% filter(!rank_score_desc_SqNoisyOr_codeLabRelativeDistance == rank_score_descUnits_SqNoisyOr_codeLabEuclidean) %>% 
            select(score_codeLab_euclidean, score_codeLab_relativeDist))


table(scores_info %>% filter(rank_score_desc < rank_score_desc_SqNoisyOr_codeLabRelativeDistance |rank_score_desc < rank_score_desc_SqNoisyOr_codeLabEuclidean) %>% select(score_0))
  
table(scores_info %>% filter(rank_score_desc < rank_score_desc_SqNoisyOr_codeLabRelativeDistance | rank_score_desc < rank_score_desc_SqNoisyOr_codeLabEuclidean) %>% select(score_1))

    