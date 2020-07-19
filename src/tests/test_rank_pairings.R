rankFile <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_scores_ranked_manually_mapped_vars.csv'




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


#check if pairing order changes score
system.time({
    combo_match <- rank_data %>% group_by(concept) %>% filter(dbGaP_studyID_datasetID_varID_2 %in% dbGaP_studyID_datasetID_varID_1) %>% 
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
