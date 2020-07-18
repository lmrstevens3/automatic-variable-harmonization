library(plyr)
library(dplyr)
library(tidyr)
library(parallel)
library(multidplyr)

#create a numeric vector from string stored in var_coding_counts_distribution_1 column of metadata
getDistDataFunction <- function(varDistString){
    var_dist = strsplit(varDistString, '; ')[[1]]
    var_dist = sort(var_dist[grep("median=[-]*[0-9]|min=[-]*[0-9]|max=[-]*[0-9]|sd=[0-9]", var_dist)])
    var_dist = setNames(gsub(".*=", "", var_dist), gsub( '=[-]*[0-9].*', "", var_dist))
    if(sum(c("max","median", "min", "sd") %in% var_dist_names) < 3){return(NULL)}
    if(sum(is.na(as.numeric(var_dist))) > 0) {warning(paste0("NA introduced in var_dist stats for: ", varDistString[is.na], " in ", varDistString))}
    var_dist[setdiff(c("max","median", "min",  "sd"),var_dist_names)] <- NA
    var_dist["range"] <- var_dist['max']-var_dist['min']
    return(var_dist)
}

create_dist_dataframe <- function(var1String, var2String){
    var1Dist = getDistDataFunction(var1String)
    var2Dist = getDistDataFunction(var2String)
    if(length(var2Dist) == 0 | length(var1Dist) == 0){return(NA)}
    return(rbind(var1Dist, var2Dist))
}

#euclideandistancefunction and relative distance functions
calc_euclidean_distance <- function(var1String, var2String){
    varsDistDf <- create_dist_dataframe(var1String, var2String)
    if(length(varsDistDf) == 0){return(NA)}
    euclideanDistance = dist(varsDistDf)
    return(euclideanDistance[[1]])
}

calc_relative_distance <- function(var1String, var2String){
    varsDistDf <- create_dist_dataframe(var1String, var2String)
    if(length(varsDistDf) == 0){return(NA)}
    relativeDistance = 1-mean(abs(apply(varsDistDf, 2, function(x){if(sum(is.na(x)) > 0){return(NA)}; if(x[[1]]-x[[2]] == 0){return(0)}; error = abs(x[[1]]-x[[2]])/max(abs(x), na.rm = T); if(error > 1){error = 1}; return(error)})), na.rm = T)
    return(relativeDistance)
}

#Function to get into each variable vs all other variables format 
calc_pairings_data <- function(pair_data, ref_data){
    pb <- txtProgressBar(min = 0, max = 6, style = 3) #track progress
    
    #replace columns with _2 to make contious vars column names reflect it is the paired variable 
    colnames(pair_data) <- gsub("_1", "_2", colnames(pair_data))
    if(!sum(grepl("_2", colnames(pair_data))) == length(colnames(pair_data))){
        warning("Some or all of continousVarData colnames did not have 1 subsituted for 2")
    }
    #get all vs all data frame
    pair_data_repeated_ref_data <- ref_data[rep(seq_len(nrow(ref_data)), each=nrow(pair_data)),]
    setTxtProgressBar(pb,1) #update progress
    pairing_data <- do.call(rbind, lapply(1:nrow(ref_data), function(x){return(pair_data)}))
    setTxtProgressBar(pb,3.5)#update progress
    filterVars.vs.continuousVarDataVars <- tbl_df(pair_data_repeated_ref_data) %>% dplyr::bind_cols(paired_data) 
    setTxtProgressBar(pb,6) #update progress
    return(filterVars.vs.continuousVarDataVars)
    close(pb)
}
 


############################################
##### main script to get continous scores

#read in all studies meta data from dbGAP var_report/xml files: 
#metaDataInfoAllStudies <- FCAM_allXMLData_NLP
var_doc_data <-read.table('/Users/laurastevens/Dropbox/Graduate School/Data Integration and Harmonization/extract_metadata/FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_noDate_noFU_5-9-19.csv', sep = ',', header = T, stringsAsFactors = F, na.strings = "")
#dim(metaDataInfoAllStudies %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1) %>% dplyr::filter(is.na(var_coding_labels_1))) 

#conceptMappedVarsData <- conceptVarsData_NLP_NAcontVar
mannual_map_var_data <- read.table('/Users/laurastevens/Dropbox/Graduate School/Data Integration and Harmonization/Manual Concept Variable Mappings BioLINCC and DbGaP/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv', sep = ',', header = T, stringsAsFactors = F, na.strings = "")

#subset data on continuous varaiables #65713 continuous variables, 565 continous variables in concept mapped variables
doc_continuous_vars <- tbl_df(var_doc_data) %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1) %>% 
    dplyr::filter(is.na(var_coding_labels_1), grepl('mean=[-]*[0-9]| median=[-]*[0-9]| min=[-]*[0-9]| max=[-]*[0-9]| sd=[-]*[0-9]', var_coding_counts_distribution_1)) %>% 
    mutate(dbGaP_studyID_datasetID_varID_1 = paste0(studyID_datasetID_1, '.', varID_1)) #create a column for variable id with study and datasett concatenated
conceptMappedVarsDataContinuous <- tbl_df(conceptMappedVarsData) %>% dplyr::filter(metadataID_1 %in% continuousVarData$metadataID_1) %>% select(-correctMatches, -data_desc_1, -detailedTimeIntervalDbGaP_1)
dim(conceptMappedVarsDataContinuous)

#65713 continuous variables, 565 continous variables in concept mapped variables takes ~4 min to get in all vs all format for scoring
continuous_var_pairings_mannual_map_var <- calc_pairings_data(doc_continuous_vars, mannual_map_var_data) %>% 
    dplyr::filter(!dbGaP_studyID_datasetID_1 == dbGaP_studyID_datasetID_2)
rm(var_doc_data, mannual_map_var_data)

#parallelize scoring process
cores <- detectCores() - 2 # mac can split 4 cores into 8 cores for parallel procesing
cluster <- makeCluster(cores)
set_default_cluster(cluster) #set default cluster so partition will use 6 cores unless specified
clusterExport(cl=cluster, list("relativeDistanceFunction", "euclideanDistanceFunction", "getDistDataFunction"),
              envir=environment())

#get scores for distributions- even with 6 cores running, takes about 2.5 hours for 37037487 rows
#scale euclidean distance score-takes about 10 minutes
#max without grouping them by studyID_datasetID_varID_1, and study_2 results in max that is 750207725, and min that is 0, which gives a lot of scores of 1
timestamp()
continuous_var_score_data <- partition(continuous_var_pairings_mannual_map_var, dbGaP_studyID_datasetID_varID_1) %>% dplyr::group_by(dbGaP_studyID_datasetID_varID_2) %>% 
    dplyr::mutate(score_euclidean_distance = calc_euclidean_distance(var_coding_counts_distribution_1, var_coding_counts_distribution_2), 
           score_relative_distance = calc_relative_distance(var_coding_counts_distribution_1, var_coding_counts_distribution_2)) %>% 
    collect() %>% group_by(dbGaP_studyID_datasetID_varID_1, study_2) %>% 
    mutate(score_euclidean_scaled = 1-((score_euclidean_distance-min(score_euclidean_distance, na.rm = T))/(max(score_euclidean_distance, na.rm = T)-min(score_euclidean_distance, na.rm = T))))
timestamp()

#stop cluster, remove large objects from memory and write data to a file-takes about 7 min
stopCluster(cluster)
rm(continuousVarDataCorrect.vs.AllContVars, doc_continuous_vars, continuous_var_pairings_mannual_map_var)
write.table(continuousVarScoreData, '/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/distribution_scores_continuous_vars.csv', sep = ',', qmethod = 'double', na = "", row.names = F)

rm(continuous_var_score_data)


