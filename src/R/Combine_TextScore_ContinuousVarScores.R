library(dplyr)
library(data.table)

allScoresFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/FHS_CHS_MESA_ARIC_all_similarity_scores_ManuallyMappedConceptVars_2.17.20.csv'
allScoresData <- fread(allScoresFile, header = T, sep = ',', na.strings =  "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()), fill = TRUE)


timestamp()
#add in noisy OR to combine scores- takes about 30 min for 21267777 rows
allScoresData <- tbl_df(allScoresData) %>% 
    mutate(score_codeLab_euclidean = replace(score_euclidianDistanceScaled, score_euclidianDistanceScaled == 0, score_codeLab[score_euclidianDistanceScaled == 0]), 
           score_codeLab_relativeDist = replace(score_relativeDistance, score_relativeDistance == 0, score_codeLab[score_relativeDistance == 0])) %>%  
    mutate(score_desc_SqNoisyOr_codeLabEuclidean = 1-((1-score_desc)^2*(1-score_codeLab_euclidean))) %>% 
    mutate(score_desc_SqNoisyOr_codeLabRelativeDistance = 1-((1-score_desc)^2*(1-score_codeLab_relativeDist))) %>%
    mutate(score_descUnits_SqNoisyOr_codeLabEuclidean = 1-((1-score_descUnits)^2*(1-score_codeLab_euclidean))) %>% 
    mutate(score_descUnits_SqNoisyOr_codeLabRelativeDistance = 1-((1-score_descUnits)^2*(1-score_codeLab_relativeDist))) %>% rowwise() %>%
    mutate(score_desc_SqNoisyOr_maxOtherMeta_euclidean = 1-((1-score_desc)^2*(1-max(c(score_units, score_codeLab, score_euclidianDistanceScaled), na.rm = T)))) %>%
    mutate(score_desc_SqNoisyOr_maxOtherMeta_relDist = 1-((1-score_desc)^2*(1-max(c(score_units, score_codeLab, score_relativeDistance), na.rm = T)))) %>% ungroup()
timestamp()


#order columns
scoreCols <- c(grep("score_",colnames(allScoresData), value = T))
allScoresData <- allScoresData[,c(colnames(allScoresData)[!colnames(allScoresData) %in% scoreCols], scoreCols)]
sapply(allScoresData, function(x){sum(is.na(x))})
#replace NAs is nes  score columns with 0 
#allScoresData <- allScoresData %>% mutate_at(scoreCols, funs(replace(., is.na(.), 0)))
fwrite(allScoresData, '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_similarity_scores_and_combined_scores_ManuallyMappedConceptVars_2.17.20.csv', sep = ',', qmethod = 'double', na = "", row.names = F)

#create a subsetted sample scores files to play with combining scores
subsetConcepts <- c("gender", "race_ethnicity",  "age", "diabetes", "smoke", "SBP", "DBP", "cholesterol_mg_dL",  "HTNmed", "BMI",  "coffee")
scoresSubsetData <- allScoresData %>% filter(conceptID %in% subsetConcepts) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1) %>% dplyr::select(-matchID, -metadataID_1, -metadataID_2)
dim(scoresSubsetData)
fwrite(scoresSubsetData, '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_similarity_scores_and_combined_scores_ManuallyMappedASCVDriskConcepts_2.17.20.csv', sep = ',', qmethod = 'double', na = "", row.names = F)
