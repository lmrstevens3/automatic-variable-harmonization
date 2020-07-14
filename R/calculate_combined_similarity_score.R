library(dplyr)
library(data.table)

allScoresFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/all_similarity_scores_manually_mapped_vars_FHS_CHS_MESA_ARIC.csv'
allScoresData <- fread(allScoresFile, header = T, sep = ',', na.strings =  "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()), fill = TRUE)

mannual_mapped_vars_file <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv'
mannual_map_vars_data <- fread(mannual_mapped_vars_file, header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))

allScoresData <- left_join(allScoresData, tbl_df(mannual_map_vars_data) %>% dplyr::select(conceptID, concept, dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_varID_1))
allScoresData <- allScoresData %>% group_by(concept) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1)

calcNoisyOR <- function(data, parent_col, child_cols){
    exp <- (1+data[[child_cols[1]]]^2+data[[child_cols[2]]]^3)
    if(length(child_cols) == 3){
        exp <- (1+data[[child_cols[1]]]^2+data[[child_cols[2]]]^3+data[[child_cols[3]]]^3)
    }
    return(1-(1-data[[parent_col]])^(exp))
}



#add in noisy OR to combine scores- takes about 30 min for 21267777 rows
combine_scores_params <- sapply(paste0("score_noisy", c("_euclidean", "_relativeDist", "_euclidean_concat", "_relativeDist_concat")), function(x){
    child_cols <- c(gsub("_concat|score_noisy", "", paste0("score_codeLab",x)), "score_units", "score_descCodeLabUnits")
    if(!grepl("_concat$", x)){child_cols = child_cols[-length(child_cols)]}
    list("data" = as.name("data"), "parent_col" = "score_desc", "child_cols" = child_cols)
}, simplify = F, USE.NAMES = T)

timestamp()
allScoresData <- tbl_df(allScoresData) %>% 
    mutate(score_codeLab_euclidean = replace(score_euclidianDistanceScaled, score_euclidianDistanceScaled == 0, score_codeLab[score_euclidianDistanceScaled == 0]), 
           score_codeLab_relativeDist = replace(score_relativeDistance, score_relativeDistance == 0, score_codeLab[score_relativeDistance == 0])) %>%  
    mutate(., !!!purrr::imap(combine_scores_params, function(combine_score_params, name, data) name = do.call("calcNoisyOR", args = combine_score_params), data = .)) %>%
    ungroup() %>% select(-score_euclidianDistance)
        
timestamp()


#order columns
score_cols <- c(grep("score_",colnames(allScoresData), value = T))
allScoresData <- allScoresData[,c(colnames(allScoresData)[!colnames(allScoresData) %in% scoreCols], scoreCols)]
sapply(allScoresData, function(x){sum(is.na(x))})
#replace NAs is nes  score columns with 0 
#allScoresData <- allScoresData %>% mutate_at(scoreCols, funs(replace(., is.na(.), 0)))
fwrite(allScoresData, '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/all_similarity_and_combined_scores_manually_mapped_vars_FHS_CHS_MESA_ARIC.csv', sep = ',', qmethod = 'double', na = "", row.names = F)

#create a subsetted sample scores files to play with combining scores
subsetConcepts <- c("gender", "race_ethnicity",  "age", "diabetes", "smoke", "SBP", "DBP", "cholesterol_mg_dL",  "HTNmed", "BMI",  "coffee")
scoresSubsetData <- allScoresData %>% filter(conceptID %in% subsetConcepts) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1) %>% dplyr::select(-matchID, -metadataID_1, -metadataID_2)
dim(scoresSubsetData)
fwrite(scoresSubsetData, '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_similarity_scores_and_combined_scores_ManuallyMappedASCVDriskConcepts_2.17.20.csv', sep = ',', qmethod = 'double', na = "", row.names = F)
