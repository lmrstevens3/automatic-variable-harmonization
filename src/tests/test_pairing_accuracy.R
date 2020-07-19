source("../my_code.R", chdir = TRUE)
library(testthat)

test_that("single number", {
    expect_equal(increment(-1), 0)
    expect_equal(increment(0), 1)
})

test_that("vectors", {
    expect_equal(increment(c(0,1)), c(1,2))
})

test_that("empty vector", {
    expect_equal(increment(c()), c())
})

test_that("test NA", {
    expect_true(is.na(increment(NA)))
})



test_mannual_map_ref_data <- fread(test_mannual_map_ref_data, "Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/src/tests/test_mannual_map_ref_data.csv", header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE)
test_scores_rank_data <- fread(test_scores_rank_data, "Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/src/R/tests/test_var_similarity_scores_rank_data.csv", header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE)


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
test_scores_rank_data <-  rank_data %>% filter(concept %in% test_concepts) %>%
    select(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_1, 
           dbGaP_studyID_datasetID_varID_2, dbGaP_studyID_datasetID_2, correctMatchTrue, 
           score_desc,  rank_score_desc, score_codeLab_relativeDist, rank_score_codeLab_relativeDist) %>% 
    unique()


test_mannual_map_ref_data <-  mannual_map_ref_data %>% filter(concept %in% test_concepts) %>%
    select(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_1, varDocID_1, study_1, 
           units_1, var_desc_1, var_coding_counts_distribution_1, possiblePairingsInConcept)

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
rank_data %>% filter(concept == test_concepts[1], (!correctMatchTrue), 
                     dbGaP_studyID_datasetID_varID_1 %in% conceptMappedVarsData[conceptMappedVarsData$concept == test_concepts[1]]$dbGaP_studyID_datasetID_varID_1 & 
                         dbGaP_studyID_datasetID_varID_2 %in% conceptMappedVarsData[conceptMappedVarsData$concept == test_concepts[1]]$dbGaP_studyID_datasetID_varID_1) %>% 
    select(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2, var_coding_counts_distribution_1, var_coding_counts_distribution_2)
rank_data %>% group_by(concept) %>%filter((!correctMatchTrue), dbGaP_studyID_datasetID_varID_2 %in% dbGaP_studyID_datasetID_varID_1)

