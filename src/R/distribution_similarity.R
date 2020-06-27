library(plyr)
library(dplyr)
library(tidyr)
library(parallel)
library(multidplyr)

#create a numeric vector from string stored in var_coding_counts_distribution_1 column of metadata
getDistDataFunction <- function(varDistString){
    varDist = strsplit(varDistString, '; ')[[1]]
    varDist = sort(varDist[grep("median=[-]*[0-9]|min=[-]*[0-9]|max=[-]*[0-9]|sd=[0-9]|nulls=[0-9]", varDist)])
    varDistStatNames = gsub( '=[0-9].*|=-[0-9].*', "", sort(varDist[grep("median=[-]*[0-9]|min=[-]*[0-9]|max=[-]*[0-9]|sd=[0-9]|nulls=[0-9]", varDist)]))
    if(sum(c("max","median", "min", "nulls", "sd") %in% varDistStatNames) < 3){return(NULL)}
    varDist = gsub(".*=", "", varDist)
    if(sum(is.na(as.numeric(varDist))) > 0) {print(paste0("NA in VarDist!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ", varDistString))}
    varDist = as.numeric(varDist)
    names(varDist) <- varDistStatNames
    varDist[!c("max","median", "min", "nulls", "sd") %in% varDistStatNames] <- NA
    varRange = varDist['max']-varDist['min']
    varDist = c(varDist, varRange)
    names(varDist) = c(c("max","median", "min", "nulls", "sd"), "range")
    return(varDist)
}



#euclideandistancefunction and relative distance functions
euclideanDistanceFunction <- function(var1String, var2String){
    var1Dist = getDistDataFunction(var1String)
    var2Dist = getDistDataFunction(var2String)
    if(length(var2Dist) == 0 | length(var1Dist) == 0){return(NA)}
    varsDistDf = rbind(var1Dist, var2Dist)
    euclideanDistance = dist(varsDistDf)
    return(euclideanDistance[[1]])
}

relativeDistanceFunction <- function(var1String, var2String){
    var1Dist = getDistDataFunction(var1String)
    var2Dist = getDistDataFunction(var2String)
    if(length(var2Dist) == 0 | length(var1Dist) == 0){return(NA)}
    varsDistDf = rbind(var1Dist, var2Dist)
    relativeDistance = 1-mean(abs(apply(varsDistDf, 2, function(x){if(sum(is.na(x)) > 0){return(NA)}; if(x[[1]]-x[[2]] == 0){return(0)}; error = abs(x[[1]]-x[[2]])/max(abs(x), na.rm = T); if(error > 1){error = 1}; return(error)})), na.rm = T)
    return(relativeDistance)
}

#Function to get into each variable vs all other variables format 
getAllvsAllDataFunction <- function(continuousVarData, filterData){
    pb <- txtProgressBar(min = 0, max = 6, style = 3) #track progress
    
    #replace columns with _2 to make contious vars column names reflect it is the paired variable 
    colnames(continuousVarData) <- gsub("_1", "_2", colnames(continuousVarData))
    if(!sum(grepl("_2", colnames(continuousVarData))) == length(colnames(continuousVarData))){
        warning("Some or all of continousVarData colnames did not have 1 subsituted for 2")
    }
    #get all vs all data frame
    continuousVarDataRepeatedfilterVars <- filterData[rep(seq_len(nrow(filterData)), each=nrow(continuousVarData)),]
    setTxtProgressBar(pb,1) #update progress
    continuousVarDataRepeated <- do.call(rbind, lapply(1:nrow(filterData), function(x){return(continuousVarData)}))
    setTxtProgressBar(pb,3.5)#update progress
    filterVars.vs.continuousVarDataVars <- tbl_df(continuousVarDataRepeatedfilterVars) %>% dplyr::bind_cols(continuousVarDataRepeated) %>% 
        dplyr::filter(!dbGaP_studyID_datasetID_1 == dbGaP_studyID_datasetID_2)
    setTxtProgressBar(pb,6) #update progress
    return(filterVars.vs.continuousVarDataVars)
    close(pb)
}
 


############################################
##### main script to get continous scores

#read in all studies meta data from dbGAP var_report/xml files: 
#metaDataInfoAllStudies <- FCAM_allXMLData_NLP
metaDataInfoAllStudies <-read.table('/Users/laurastevens/Dropbox/Graduate School/Data Integration and Harmonization/extract_metadata/FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_noDate_noFU_5-9-19.csv', sep = ',', header = T, stringsAsFactors = F, na.strings = "")
#dim(metaDataInfoAllStudies %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1) %>% dplyr::filter(is.na(var_coding_labels_1))) 

#conceptMappedVarsData <- conceptVarsData_NLP_NAcontVar
conceptMappedVarsData <- read.table('/Users/laurastevens/Dropbox/Graduate School/Data Integration and Harmonization/Manual Concept Variable Mappings BioLINCC and DbGaP/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv', sep = ',', header = T, stringsAsFactors = F, na.strings = "")

#subset data on continuous varaiables #65713 continuous variables, 565 continous variables in concept mapped variables
continuousVarData <- tbl_df(metaDataInfoAllStudies) %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1) %>% 
    dplyr::filter(is.na(var_coding_labels_1), grepl('mean=[-]*[0-9]| median=[-]*[0-9]| min=[-]*[0-9]| max=[-]*[0-9]| sd=[-]*[0-9]', var_coding_counts_distribution_1)) %>% 
    mutate(dbGaP_studyID_datasetID_varID_1 = paste0(studyID_datasetID_1, '.', varID_1)) #create a column for variable id with study and datasett concatenated
conceptMappedVarsDataContinuous <- tbl_df(conceptMappedVarsData) %>% dplyr::filter(metadataID_1 %in% continuousVarData$metadataID_1) %>% select(-correctMatches, -data_desc_1, -detailedTimeIntervalDbGaP_1)
dim(conceptMappedVarsDataContinuous)


#65713 continuous variables, 565 continous variables in concept mapped variables takes ~4 min to get in all vs all format for scoring
continuousVarDataCorrect.vs.AllContVars <- getAllvsAllDataFunction(continuousVarData, filterData = conceptMappedVarsDataContinuous)
rm(metaDataInfoAllStudies, conceptMappedVarsData)


#parallelize scoring process
cores <- detectCores() - 2 # mac can split 4 cores into 8 cores for parallel procesing
cluster <- makeCluster(cores)
set_default_cluster(cluster) #set default cluster so partition will use 6 cores unless specified
clusterExport(cl=cluster, list("relativeDistanceFunction", "euclideanDistanceFunction", "getDistDataFunction"),
              envir=environment())

#get scores for distributions- even with 6 cores running, takes about 2.5 hours for 37037487 rows
timestamp()
continuousVarScoreData <- partition(continuousVarDataCorrect.vs.AllContVars, dbGaP_studyID_datasetID_varID_1) %>% dplyr::group_by(dbGaP_studyID_datasetID_varID_2) %>% 
    dplyr::mutate(score_euclideanDistance = euclideanDistanceFunction(var_coding_counts_distribution_1, var_coding_counts_distribution_2), 
           score_relativeDistance = relativeDistanceFunction(var_coding_counts_distribution_1, var_coding_counts_distribution_2)) 
timestamp()

#scale euclidean distance score-takes about 10 minutes
#max without grouping them by studyID_datasetID_varID_1, and study_2 results in max that is 750207725, and min that is 0, which gives a lot of scores of 1
continuousVarScoreData <- continuousVarScoreData %>% collect() %>% group_by(dbGaP_studyID_datasetID_varID_1, study_2) %>% 
    mutate(score_euclideanDistanceScaled = 1-((score_euclideanDistance-min(score_euclideanDistance, na.rm = T))/(max(score_euclideanDistance, na.rm = T)-min(score_euclideanDistance, na.rm = T))))

#stop cluster, remove large objects from memory and write data to a file-takes about 7 min
stopCluster(cluster)
rm(continuousVarDataCorrect.vs.AllContVars, continuousVarData, conceptMappedVarsDataContinuous)
write.table(continuousVarScoreData, '/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/ContinuousVarsDistributionScores_7-17-19.csv', sep = ',', qmethod = 'double', na = "", row.names = F)

rm(continuousVarScoreData)


