library(data.table)
library(dplyr)

rank_scores <- function(data, score_col, ref_ID_col, rank_by_col = NULL){
    #' sorts data by score_col (descending) and dense ranks pairings with highest similarity scores. 
    #' Pairings with a score of zero recieve a rank value of NA. A rank_<score_col> column is appended to data 
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param score_col: the column name of the column in data containing similarity scores
    #' @param ref_ID_col: the column in data that contains unique ref IDs
    #' @rank_by_col: An optional grouping variable that defines if ranking 1:n should be done by group instead of for all pairings for a given ref ID. 
    #' For example, if scores for a ref come from multiple sources, the column listing the scores source could be supplied, 
    #' and ref ID would be ranked 1:n scores for each source instead of 1:n scores in all sources
    rankData <- data %>% group_by_at(c(ref_ID_col, rank_by_col), add = T) %>% 
        arrange(desc(!!as.name(score_col)), .by_group = T)  %>% 
        mutate(!!as.symbol(paste0("rank_",score_col)) := dense_rank(desc(!!as.name(score_col)))) %>% 
        mutate_at(paste0("rank_",score_col), funs(replace(., !!as.name(score_col) == 0, NA))) %>% 
        group_by(!!as.name(ref_ID_col)) %>% 
        mutate(!!as.symbol(paste0(score_col, "_ranked_n")) := sum(!is.na(!!as.name(paste0("rank_",score_col)))))
    
    rankData <- rankData %>% dplyr::select(grep(paste0(score_col,"$"), colnames(.), invert = T), matches(score_col,"$"))
    return(rankData)
}

calc_ranks_all_scores <- function(data, score_variables, ref_ID_col, rank_by_col = NULL, parallel = F, cores = (detectCores() - 2)){
    #wrapper function to parallelize or append a set of rank cols using rank_scores function for multiple pairing scores
    #' 
    #' @param data: dataframe of pairings with a column of similarity scores used to rank pairings
    #' @param score_variables: vector of the score column names of the columns in data containing similarity scores to be ranked
    #' @param ref_ID_col: the column in data that contains unique ref IDs
    #' @rank_by_col: An optional grouping variable that defines if ranking 1:n should be done by group instead of for all pairings for a given ref ID. 
    #' see rank_scores for additional details
    #' @parallel: if True, code will be parallelized for optmization 
    #' @cores: number of cores to use for parallel processing (default is cores available - 2)
    if (parallel) {
        set_default_cluster(makeCluster(cores))
        data <- data %>% multidplyr::partition(., !!as.name(ref_ID_col))
    }
    for(i in score_variables) {
        print(paste0("calculating ", i, " ranks"))
        data <- rank_scores(data, i, ref_ID_col, rank_by_col)
    }
    if(parallel) {
        data <- data %>% multidplyr::collect(.)
        on.exit(stopCluster(cluster)) 
    }
    return(data)
}

##DO ALL SCORE MERGING AND CONCEPT CHECKING PRIOR TO RANKING
mannual_map_ref_file = '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv'

scores_file <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/output/all_similarity_scores_manually_mapped_vars_FHS_CHS_MESA_ARIC.csv'
rank_output_file <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_scores_ranked_manually_mapped_vars.csv'


#all_similarity_and_combined_scores_manually_mapped_vars_FHS_CHS_MESA_ARIC.csv
mannual_map_ref_data <- fread(mannual_map_ref_file, header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))
mannual_map_ref_data <- mannual_map_ref_data %>% group_by(concept) %>% 
    filter(!length(dbGaP_studyID_datasetID_varID_1) == 1) %>%
    rename("var_docID_1" = varDocID_1) 

scores_data <- fread(scores_file, header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))

#scores_data <- scores_data %>% select("var_docID_1" = metadataID_1, "var_docID_2" = metadataID_2, 
                                      #dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2, 
                                      #dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_2, matches("score_"))
dim(scores_data)
colnames(scores_data)
#Join scores data and concept data and filter out datasets not in goldstandard and concepts with broad definitions that don't map across variables/studies (ex. cholesterol lowering med because not same medication for mappings)
#takes about 3-4 min with 21267777 X 41 and 1703 X 23
scores_data <- dplyr::full_join(scores_data, mannual_map_ref_data %>% select(concept, var_docID_1, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_1)) %>% 
    filter(!concept %in% c("Cholesterol Lowering Medication")) %>%
    group_by(concept) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1 | is.na(dbGaP_studyID_datasetID_varID_2)) 

#need to decide how to handle vars that get filtered because no pairs with score > 0 
gsub('([a-z])\\B([A-Z])', '\\1_\\L\\2', gsub('odeLab', 'odelab', colnames(scores_data)[-c(1:6)]), perl=T)
#"unique Participant Identifier" (could also remove if desired)
#check join worked correctly
unique(scores_data$concept) #50 concepts, 0.99 GB, 2367346 rows after merge and filter (#51 concepts, 9218935 rows, 4.2 GB if use PID concept) (#concepts Time To CVDDeath and Time To CHDDeath filtered out because they are only present in mesa)
dim(scores_data)
#check vars that are in ref data but not in scores (either not scored, score of 0 or purposefully removed)
mannual_map_ref_data %>% filter(!dbGaP_studyID_datasetID_varID_1 %in% unique(scores_data$dbGaP_studyID_datasetID_varID_1),
                                !concept %in% c("Cholesterol Lowering Medication")) %>% 
    select(concept, study_1, dbGaP_dataset_label_1, dbGaP_studyID_datasetID_varID_1, var_desc_1)

#calculate ranks
score_cols <- c(grep("^score_",colnames(scores_data), value = T))
rank_data <- calc_ranks_all_scores(scores_data, score_cols, ref_ID_col = "dbGaP_studyID_datasetID_varID_1", rank_by_col = "dbGaP_studyID_datasetID_2")


fwrite(rank_data, rank_output_file, sep = ',', qmethod = 'double', na = "", row.names = F)







    