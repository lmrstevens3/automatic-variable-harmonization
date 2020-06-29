library(dplyr)
library(data.table)

#filePaths
allTextScoreDataFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/tiff_laura_shared/NLP text Score results/FHS_CHS_MESA_ARIC_text_similarity_scores_All_Scores_ManuallyMappedConceptVars_7.17.19.csv'
metadataInfoAllStudiesFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/tiff_laura_shared/FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_5-9-19.csv'
continuousVarScoreDataFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/ContinuousVarsDistributionScores_7-17-19.csv'

#takes approximately 2 min to load
allTextScoreData <- fread(allTextScoreDataFile, header = T, sep = ',', na.strings =  "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()), fill = TRUE)
dim(allTextScoreData)
allTextScoreData <- tbl_df(allTextScoreData) %>% dplyr::mutate(dbGaP_studyID_datasetID_varID_1 = paste0(studyID_datasetID_1,'.', varID_1), dbGaP_studyID_datasetID_varID_2 = paste0(studyID_datasetID_2,'.', varID_2)) %>% 
    dplyr::filter(studyID_datasetID_2 %in% studyID_datasetID_1) %>% dplyr::rename(dbGaP_studyID_datasetID_2 = studyID_datasetID_2, dbGaP_studyID_datasetID_1 = studyID_datasetID_1)

#add in extra meta data info for matched variables
metadataInfoAllStudies <- fread(metadataInfoAllStudiesFile, header = T, sep = ',', stringsAsFactors=FALSE, na.strings =  "", showProgress=getOption("datatable.showProgress", interactive()))
allTextScoreData <- allTextScoreData %>% dplyr::left_join(tbl_df(metadataInfoAllStudies) %>% dplyr::select(-data_desc_1, -detailedTimeIntervalDbGaP_1) %>% dplyr::rename(dbGaP_studyID_datasetID_1 = studyID_datasetID_1))
matchedMetaDataInfoAllStudies <- metadataInfoAllStudies %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1) %>% dplyr::rename(dbGaP_studyID_datasetID_1 = studyID_datasetID_1)
colnames(matchedMetaDataInfoAllStudies) <- gsub("_1", "_2", colnames(matchedMetaDataInfoAllStudies))
allTextScoreData <- allTextScoreData %>% dplyr::left_join(matchedMetaDataInfoAllStudies)

dim(allTextScoreData)
sapply(allTextScoreData, function(x){sum(is.na(x))})
rm(matchedMetaDataInfoAllStudies, metadataInfoAllStudies)

#add in continuousVarsData
continuousVarScoreData <- fread(continuousVarScoreDataFile, header = T, sep = ',', stringsAsFactors=FALSE, na.strings = "", showProgress=getOption("datatable.showProgress", interactive()))
continuousVarScoreData <- continuousVarScoreData %>% dplyr::mutate_if(is.logical, as.character) %>% 
    dplyr::rename("dbGaP_studyID_datasetID_2" = "studyID_datasetID_2", "dbGaP_studyID_datasetID_1"= "studyID_datasetID_1","score_relativeDistance" = "relativeDistanceScore", "score_euclidianDistanceScaled" = "euclidianDistanceScoreScaled", "score_euclidianDistance" = "euclidianDistanceScore") %>% 
    dplyr::select(-concept, -conceptID)#with 21267777 X 26  and 37037487 X 27 takes about 4 min
allScoresData <- dplyr::left_join(allTextScoreData, continuousVarScoreData)

dim(allScoresData)
sapply(allScoresData, function(x){sum(is.na(x))}) #all variables should have 0 NA except dbGaP_cohort, score vars, timeIntervalDbGaP, units, var_coding_labels


#replace scores with NA with a score of 0 prior to calculating distance scores
scoreCols <- c(grep("score_",colnames(allScoresData), value = T))
allScoresData <- allScoresData %>% dplyr::select(-var_coding_labels_1, -var_coding_labels_2) %>% mutate_at(scoreCols, funs(replace(., is.na(.), 0)))
sapply(allScoresData, function(x){sum(is.na(x))})
rm(allTextScoreData, continuousVarScoreData)

fwrite(allScoresData, '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_similarity_scores_ManuallyMappedConceptVars_2.17.20.csv', sep = ',', qmethod = 'double', na = "", row.names = F)

