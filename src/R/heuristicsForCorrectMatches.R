#get the top 2 matches for every variable and calculate the ratio between the top two matches' scores
topScoresRatioFunction <- function(data, variable){
    data$score <- data[[variable]]
    data$rank <- data[[paste0("rank_", variable)]]
    top2scores <- data %>% group_by(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% filter(rank == 1 | rank == 2) %>% 
        mutate(correctMatchTrueRank = ifelse(correctMatchTrue, rank, 0)) %>% 
        select(study_1, dbGaP_studyID_datasetID_varID_1, study_2,  dbGaP_studyID_datasetID_varID_2, reference_variable, matched_variable, rank, score, correctMatchTrue, correctMatchTrueRank)
    scoreRatioData <- top2scores %>% mutate(topScore = unique(score[rank==1]), correctMatchRank1 = max(ifelse(correctMatchTrueRank == 1, 1, 0)), 
                                            secondScore = ifelse(n() == 2, unique(score[rank==2]), NA), correctMatchRank2 = max(ifelse(correctMatchTrueRank == 2, 1, 0)), 
                                            ratio = ifelse(n() == 2, score[rank==1]/score[rank == 2], NA)) ungroup()
    return(scoreRatioData)
}

percentTopScoreFunction <- function(data,variable){
    data$score <- data[[variable]]
    data$rank <- data[[paste0("rank_", variable)]]
    percentScoresData <- data %>% group_by(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% mutate('percentTopScore' = score/(unique(score[rank == 1]))) %>% select(concept, study1_id,  study1_dbGaP_dataset_name, studyID_datasetID_2,  study2_dbGaP_dataset_name, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2, reference_variable, matched_variable, percentTopScore, rank, correctMatchTrue, score)
    return(percentScoresData)
}

rankDataFile <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_scores_ranked_ManuallyMappedConceptVars2.csv'
rankData <- fread(rankDataFile , header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))

scoreRatioDataNoisyOr <- topScoresRatioFunction(rankData ,"score_desc_SqNoisyOr_maxOtherMeta_euclidean")
plot_ly(scoreRatioDataNoisyOr, x = ~topScore, y = ~log(ratio), color = ~as.factor(correctMatchRank1), type = 'scatter', mode = "markers")

#create heat map of F1 score
applyQuantilesFunction <- function(){}
F1scoreHeuristicsData <- scoreRatioData %>% mutate(bin_topScoreRatio = applyQuantiles(ratio, 10), bin_scoreThreshold = applyQuantiles(score, 10)) %>% 
    select(bin_topScoreRatio, bin_scoreThreshold"(for rank 1 only)", correctMatchRank1) %>% group_by(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% 
    group_by(bin_topScoreRatio, bin_scoreThreshold) %>% summarize(ratio_threshold_recall = sum(unique(correctMatchRank1))/length(unique(correctMatchRank1[rank == 1])))
#"recall" = sum(unique(grouped by var, dataset2, correctMatchTrue))/length(unique(grouped by var, dataset2))
F1scoreHeuristicsHeatmapData <- F1scoreHeuristicsData %>% spread(bin_scoreThreshold, ratio_threshold_recall)
pheatmap(as.matrix(F1scoreHeuristicsHeatmapData), clusterrows = F, clustercols = F)

#check rank Data and rank Differences
timestamp()
decreased_scores <- rankData %>% 
    group_by(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% 
    filter(sum(correctMatchTrue) > 0) %>% ungroup() %>% rowwise() %>%
    filter(!sum(is.na(c(rank_score_desc,  
                       rank_score_desc_SqNoisyOr_maxOtherMeta_euclidean, 
                       rank_score_desc_SqNoisyOr_codeLabRelativeDistance, 
                       rank_score_desc_SqNoisyOr_codeLabEuclidean, 
                       rank_score_descCodeLabUnits)), na.rm = T) == 5) %>% 
    mutate(max_rank = max(c(rank_score_desc,  
                            rank_score_desc_SqNoisyOr_maxOtherMeta_euclidean, 
                            rank_score_desc_SqNoisyOr_codeLabRelativeDistance, 
                            rank_score_desc_SqNoisyOr_codeLabEuclidean)), 
           min_rank = min(c(rank_score_desc,  
                            rank_score_desc_SqNoisyOr_maxOtherMeta_euclidean, 
                            rank_score_desc_SqNoisyOr_codeLabRelativeDistance, 
                            rank_score_desc_SqNoisyOr_codeLabEuclidean))) %>% ungroup() %>%
    group_by(concept, dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% 
    mutate(correctMatch_max_rank = max(unique(max_rank[correctMatchTrue == T]))) %>%
    ungroup() %>% filter(max_rank <= correctMatch_max_rank) %>% select(concept, 
                       reference_variable, matched_variable, 
                       dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_2,  
                       rank_score_desc,  
                       rank_score_desc_SqNoisyOr_maxOtherMeta_euclidean, 
                       rank_score_desc_SqNoisyOr_codeLabRelativeDistance, 
                       rank_score_desc_SqNoisyOr_codeLabEuclidean, 
                       rank_score_descCodeLabUnits, 
                       max_rank, correctMatch_max_rank, min_rank, correctMatchTrue, 
                       var_coding_counts_distribution_1, var_coding_counts_distribution_2,
                       score_desc, score_descCodeLabUnits, score_units, 
                       score_codeLab_relativeDist, score_codeLab, score_codeLab_euclidean, 
                       score_desc_SqNoisyOr_codeLabRelativeDistance, score_desc_SqNoisyOr_codeLabEuclidean,
                       dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2)
timestamp()

#check units similarity for variables that aren't continuous
continuous_unit_pairings <- rankData %>% dplyr::filter(score_units > 0) %>%
    dplyr::filter(grepl('(?:max|min|mean|median|sd)[ ]*=[ ]*(?:\\d|.\\d)*;+', var_coding_counts_distribution_1) & 
                      grepl('(?:max|min|mean|median|sd)[ ]*=[ ]*(?:\\d|.\\d)*;+', var_coding_counts_distribution_2)) %>%
    dplyr::select(reference_variable, matched_variable,  
                  score_units, units_1, units_2, 
                  score_codeLab, var_coding_counts_distribution_1, var_coding_counts_distribution_2)

#95% of pairings with similarity score for units are continuous variables (98% if score_desc > 0)
summary_rankData <- head(rankData, 10) %>% dplyr::filter(score_desc > 0) %>% 
    dplyr::select(concept,reference_variable, matched_variable,
           matches("^(?:score_|rank_)"), correctMatchTrue, 
           units_1, units_2, 
           var_coding_counts_distribution_1, var_coding_counts_distribution_2,
           dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_2,
           dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2) %>%
    mutate(ref_var_continuous = grepl('(?:max|min|mean|median|sd)[ ]*=[ ]*(?:\\d|.\\d)*;+', var_coding_counts_distribution_1), 
           paired_var_continuous = grepl('(?:max|min|mean|median|sd)[ ]*=[ ]*(?:\\d|.\\d)*;+', var_coding_counts_distribution_2)) %>%
    mutate(pairing_data_type = rowSums(select(.,"ref_var_continuous", "paired_var_continuous")),
           pairing_data_type = plyr::mapvalues(pairing_data_type, c(0,1,2), c("categorical", "categorical_continous", "continous"))) %>%
    tidyr::gather(key = noisy_score_type, value = noisy_score, -c(matches("ID|var|units|correctMatchTrue|^rank_|^score_(?:![\\w_]*_SqNoisyOr_)")))%>%
    #group_by( dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2) %>% 
    mutate(noisy_all_eql  = length(unique(noisy_score) == 0), 
           noisy_scores  = gsub("\\|score_units((?:E|R))", "\\L\\1", 
                                gsub("_SqNoisyOr(?:_codeLab|_maxOtherMeta_\\w+$)", "|score_codeLab_|score_units",noisy_score_type))) 
#add more things
summary_rankData <- summarise_(total_pairs = n(), 
               continous_pairs = sum(ref_var_continous & paired_var_continous), 
               continuous_categorical_pairs = sum(!ref_var_continous == paired_var_continous), 
               categorical_pairs = sum(!(ref_var_continous & paired_var_continous))) 
               
               
    


#

#test creating correctMatchTrue Variable
dbp_t <- rankData %>% filter(conceptID == "DBP") %>% left_join(conceptMappedVarsData)
system.time({
    match_test1 <- dbp_t %>% group_by(dbGaP_studyID_datasetID_varID_1,  dbGaP_studyID_datasetID_varID_2) %>% 
    dplyr::mutate(correctMatchTrue_test1 = isCorrectMatch(dbGaP_studyID_datasetID_varID_2, possiblePairingsInConcept))
})

system.time({
    match_test2 <- dbp_t %>% group_by(concept) %>%  
        mutate(correctMatchTrue_test2 = dbGaP_studyID_datasetID_varID_2 %in% dbGaP_studyID_datasetID_varID_1) 
})

system.time({
    match_test3 <- dbp_t %>% group_by(dbGaP_studyID_datasetID_varID_1,  dbGaP_studyID_datasetID_varID_2) %>% 
        tidyr::separate_rows(possiblePairingsInConcept, sep = "; ", convert = FALSE) %>% 
        group_by(concept) %>%  
        mutate(correctMatchTrue_test3 = dbGaP_studyID_datasetID_varID_2 %in% possiblePairingsInConcept) 
})

#filter(concept %in% c("alcohol intake (gm)", "Coffee Intake (cups/day)", "sodium intake (mg)", "stroke", "systolic murmur (grade)"), correctMatchTrue) 


#look a variable distribution scores differnences when using euclidean vs. relative distance
library(plotly)
plot_ly(x = rankData$score_euclidianDistanceScaled, y = rankData$score_relativeDistance, type = 'scatter', mode = "markers")
contScoresBox <- plot_ly(y = ~rankData$score_relativeDistance, type = "box") %>% add_trace(y = ~rankData$score_euclidianDistanceScaled)
contScoresBox


plot_ly(x = alch_scores$score_varDescOnly, y = alch_scores$allText_SqNoisyOr_MetricsError, color = alch_scores$rankDiff_desc_noisyOR, type = 'scatter', mode = "markers")






write.table(scoreRatioDataNoisyOr, '/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/scoreRatioData_varDescOnly_score.csv', sep = ',', row.names = F, qmethod = 'double', na = "")
write.table(scoreRatioDataNoisyOr2, '/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/scoreRatioData_varDescOnly_allTextIf0.csv', sep = ',', row.names = F, qmethod = 'double', na = "")
write.table(scorePercentTopDataNoisyOr, '/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/scorePercentTop_varDescOnly_noisyOr_allText_SqNoisyOr_MetricsError.csv', sep = ',', row.names = F, qmethod = 'double', na = "")
write.table(scorePercentTopDataNoisyOr2, '/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/scorePercentTop_varDescOnly_allTextIf0.csv', sep = ',', row.names = F, qmethod = 'double', na = "")

#filtering out var comparisons in same dataset (assuming each var is unique and no need to compare within datasets)
checkIncorrectRatioHigh <- scoreRatioDataNoisyOr2 %>% filter(ratio > exp(1) & (correctMatchRank1 < 1))
checkIncorrectRatioHigh %>% select(topScore, ratio, ref_var, correctMatchTrueRatio, ratioTop2Matches)

displayDataTables <- sapply(names(ranksData), function(x){displayDataCorrectMatches(ranksData[[x]], x)}, USE.NAMES = T, simplify = F)

getColumnColors <- function(displayData){
    rankCols <- grepl('_Rank',colnames(x))
    possibleRanks <- na.omit(unique(unlist(x[,rankColumn])))
    #cols_1match <- "rgb(255, 0, 255)"
    clrs_NAmatch <- "rgb(255, 255, 255)"
    clrs_1match <- "rgb(255, 0, 255)"
    clrs_2_5match <- rep("#7EDCF1", length(possibleRanks[possibleRanks > 1 & possibleRanks <=5]))
    clrs_5_10match <- rep("#D086F1", length(possibleRanks[possibleRanks > 5 & possibleRanks <= 10]))
    clrs_11match <- rep("#F71D34", length(possibleRanks[possibleRanks > 10]))
    #clrs_2_5match <- round(seq(255, 60, length.out = length(possibleRanks[possibleRanks > 1 & possibleRanks <=5])), 0) %>% {paste0("rgb(", . , ", ", . , "255)")}
    #clrs_5_10match <- round(seq(255, 60, length.out = length(possibleRanks[possibleRanks > 5 & possibleRanks <= 10])), 0) %>% {paste0("rgb(", ., ", 255,", ., ")")}
    #clrs_11match <- round(seq(255, 60, length.out = length(possibleRanks[possibleRanks > 10])), 0) %>% {paste0("rgb(255, ", . , ", ", . , ")")}
    clrs <- c(clrs_NAmatch, clrs_2_5match, clrs_5_10match, clrs_11match)
    names(clrs) <- c("NA", sort(possibleRanks), decreasing = T)
    return(clrs)
}

sapply(displayDataTables, function(displayData){
    
    clrs <- getColumnColors(displayData)
    datatable(displayData, options = list(
        columnDefs = list(list(targets = c(7:14), visible = FALSE)))) %>%
        formatStyle("ARIC", "ARIC_Rank", backgroundColor = styleEqual(unique(displayData[["ARIC_Rank"]]), clrs[as.character(unique(displayData[["ARIC_Rank"]]))])) %>% formatStyle("CHS", "CHS_Rank", backgroundColor = styleEqual(unique(displayData[["CHS_Rank"]]), clrs[as.character(unique(displayData[["CHS_Rank"]]))])) %>% 
        formatStyle("FHS", "FHS_Rank", backgroundColor = styleEqual(unique(displayData[["FHS_Rank"]]), clrs[as.character(unique(displayData[["FHS_Rank"]]))])) %>% formatStyle("MESA", "MESA_Rank", backgroundColor = styleEqual(unique(displayData[["MESA_Rank"]]), clrs[as.character(unique(displayData[["MESA_Rank"]]))]))
})




displayDataCorrectMatches <-function(ranksData, scoreVariable){
    ranksData$score <- ranksData[[scoreVariable]]
    ranksSubset <- ranksDescUnitsCodeVal %>% dplyr::filter(correctMatchTrue == T) %>% group_by(dbGaP_studyID_datasetID_varID_1) %>% mutate(matched_variable.score.rank = paste0(unique(c(matched_variable, score, rank)), collapse = ": "))
    
    spreadStudy2Var <- ranksSubset %>% group_by(concept, dbGaP_studyID_datasetID_varID_1, study2) %>% spread(matched_variable.score.rank) %>% group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% select(concept, study1, dbGaP_studyID_datasetID_varID_1, study1_var_id, study1_var_name, reference_variable, ARIC, CHS, FHS, MESA)
    gatherStudy2Var <- gather(spreadStudy2Var, matched_variable.score.rank, -concept, -study1, -study1_var_id, -study1_var_name, -dbGaP_studyID_datasetID_varID_1, -reference_variable) %>% na.omit()
    spreadStudy2VarMatrix <- gatherStudy2Var %>% group_by(concept, reference_variable, study2) %>% mutate(study2VarNum = row_number()) %>% spread(matched_variable.score.rank)
    
    spreadRanksStudy2Rank <- ranksSubset %>% group_by(concept, dbGaP_studyID_datasetID_varID_1, study2) %>% spread(study2, rank) %>% group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% select(concept, study1, dbGaP_dataset_name, study1_id, dbGaP_studyID_datasetID_varID_1, study1_var_id, study1_var_name, reference_variable, ARIC, CHS, FHS, MESA)
    gatherRanksStudy2Rank <- gather(spreadRanksStudy2Rank, study2, rank, -concept, -study1, -study1_id, -dbGaP_dataset_name, -study1_var_id, -study1_var_name, -dbGaP_studyID_datasetID_varID_1, -reference_variable) %>% na.omit()
    spreadStudy2VarRanksMatrix <- gatherRanksStudy2Rank %>% group_by(concept, reference_variable, study2) %>% mutate(study2VarNum = row_number()) %>% spread(study2, rank)
    
    spreadStudy2Score <- ranksSubset %>% group_by(concept, dbGaP_studyID_datasetID_varID_1, study2) %>% spread(study2, score) %>% group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% select(concept, study1, dbGaP_dataset_name, study1_id, dbGaP_studyID_datasetID_varID_1, study1_var_id, study1_var_name, reference_variable, ARIC, CHS, FHS, MESA)
    gatherStudy2Score <- gather(spreadStudy2Score, study2, score, -concept, -study1, -study1_id, -dbGaP_dataset_name, -study1_var_id, -study1_var_name, -dbGaP_studyID_datasetID_varID_1, -reference_variable) %>% na.omit()
    spreadStudy2VarScoreMatrix <- gatherStudy2Score %>% group_by(concept, reference_variable, study2) %>% mutate(study2VarNum = row_number()) %>% spread(study2, score)
    
    heatMapRanksMatrix <- spreadStudy2VarRanksMatrix %>% ungroup() %>% select(concept, reference_variable, ARIC, CHS, FHS, MESA)
    heatMapVarMatrix <- spreadStudy2VarMatrix %>% ungroup() %>% select(concept,  reference_variable, ARIC, CHS, FHS, MESA)
    heatMapScoreMatrix <- spreadStudy2VarScoreMatrix %>% ungroup() %>% select(concept, reference_variable, ARIC, CHS, FHS, MESA)
    
    heatMapVarMatrixRanksScore <- bind_cols(heatMapVarMatrix, heatMapRanksMatrix[,c("ARIC", "CHS", "FHS", "MESA")], heatMapScoreMatrix[,c("ARIC", "CHS", "FHS", "MESA")])
    colnames(heatMapVarMatrixRanksScore) <- c(colnames(heatMapVarMatrix), paste0(c("ARIC", "CHS", "FHS", "MESA"), paste0('_Rank', scoreVariable)), paste0(c("ARIC", "CHS", "FHS", "MESA"), paste0('_', scoreVariable)))
    
}




###Gold Standard
gs <- unique(rankData2 %>% dplyr::filter(correctMatchTrue == T) %>% group_by(concept, dbGaP_studyID_datasetID_varID_1) %>% mutate(dataset_variable =  paste0(unique(c(study1_dbGaP_dataset_name, study1_var_id, study1_var_name)), collapse = ": ")) %>% select(study1,  dataset_variable))
gs2 <- tidyr::gather(gsByStudy, study, dataset_variable, -concept) %>% na.omit()
gsByStudy <- gs %>% group_by(concept) %>% tidyr::spread(study1, dataset_variable) %>% select(-dbGaP_studyID_datasetID_varID_1) 

#take diabetes, BMI, HDL, Race/Ethnicity
datatable(gsByStudy[gsByStudy$concept %in% c("Race/Ethnicity", "Systolic Blood Pressure (mmHg)"),c(1,5,3,2,4)], escape = FALSE)



### plot rank differences for every variable with cummulative change curve for all plot
#plot_ly(t, type = 'histogram')
par(mar = c(5,5,2,5))
rankDiffVector <- rankDifferences$descriptionCode_TextScore.RANK_DIFF.description_TextScore
par(new = T)
h <- hist(
    rankDiffVector,
    breaks = c(min(rankDiffVector) - 1, sort(unique(rankDiffVector))), 
    xlim = c(min(rankDiffVector) - 1,20))

cumCounts <- ecdf(rankDiffVector)


barplot(rankCounts[,2], main="Rank Differences From Desc Text to Desc+Coding Values", 
        xlab="Rank Difference", axes=F, ylab=NA, col = rgb(0,0,0,alpha=0))

plot(x = sort(unique(rankDiffVector)), y = sort(unique())*length(rankDiffVector), type = 'n', col = 'red')
hist
axis(4, at=seq(from = 0, to = length(rankDiffVector), length.out = 11), labels= seq(0, 1, 0.1), col = 'red', col.axis = 'red')
lines(x = sort(unique(rankDiffVector)), y = sort(unique(cumCounts))*length(rankDiffVector),  col = 'red')
mtext(side = 4, line = 3, 'Cumulative Density', col = 'red')



rankCounts <- cbind.data.frame("rankDiff" = sort(unique(rankDiffVector)), "count" = as.vector(table(rankDiffVector)), "cumulativeDensity"  = sort(unique(cumCounts(rankDiffVector)))*max(as.vector(table(rankDiffVector))))
plot_ly(rankCounts) %>% add_trace(x = ~rankDiff, y = ~count, type = 'bar', name = 'Count') %>% add_trace(x= ~rankDiff, y = ~cumulativeDensity, type = 'scatter', mode = 'lines', name = 'Density') %>% 
    layout(        
        xaxis = list(title = "rank difference"), 
        yaxis = list(title = "paired variables rank difference frequency", showgrid = FALSE, zeroline = FALSE), 
        yaxis2 = list(overlaying = "y", side = "right", zeroline = F, showgrid = FALSE, title = "Cummulative Density", autoRange = T, range = c(0, 1)),
        showlegend = T)



htplot_ly(rankDifferencesdescription_codeVal_TextScore_rank.DIFF_FROM.description_TextScore_rank, type = "histogram", cumulative = list(enabled=TRUE), fill = "tozeroy", yaxis = "y2", name = "Density") 
plot_ly(rankDifferences2, x = ~description_codeVal_TextScore_rank.DIFF_FROM.description_TextScore_rank, type = 'histogram')  






#examples with colorspace colors to create color scales
library(colorspace)
R> pal <- function(col, border = "light gray", ...){
    n <- length(col)
    plot(0, 0, type="n", xlim = c(0, 1), ylim = c(0, 1),
         axes = FALSE, xlab = "", ylab = "", ...)
    rect(0:(n-1)/n, 0, 1:n/n, 1, col = col, border = border)
}

R> pal(diverge_hcl(7))
R> pal(diverge_hcl(7, c = 100, l = c(50, 90), power = 1))
R> pal(diverge_hcl(7, h = c(130, 43), c = 100, l = c(70, 90)))
R> pal(diverge_hcl(7, h = c(180, 330), c = 59, l = c(75, 95)))



#check Rankings are correct using dplyr 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# match1 = 0
# match5 = 0
# match10 = 0
# numberOfMatches = 0
# targetConcept <-"Marital Status"# tc#"Current Smoking Status (smoker in last year)"
# correctConceptSubset <- tbl_df(correctVarsData)[correctVarsData$concept == targetConcept,]  %>% select(concept, dbGaP_studyID_datasetID_varID_1, correctMatches)
# correctMatchList <- strsplit(unique(correctConceptSubset[,"correctMatches"])[[1]], "; ")[[1]]
# var1 = 'phs000280.v3.pht004062.v1.DIABTS22'#correctMatchList[1]
# dataset1 = gsub('^(.+)\\..+$',"\\1",var1)
# var1name =  gsub('^.+\\.(.+)$',"\\1",var1)
# var2 = setdiff(correctMatchList, var1)[1]
# dataset2 = gsub('^(.+)\\..+$',"\\1",var2)
# var2name = gsub('^.+\\.(.+)$',"\\1",var2)
# dataset2Matches <- textScoreDataDescOnly %>% filter((study1_id == dataset1) & (study1_var_id == var1name) & (studyID_datasetID_2 == dataset2)) %>% arrange(desc(description_TextScore))
# var2Rank <- ifelse(var2name %in% dataset2Matches$study2_var_id ,dataset2Matches$rank[match(var2name,dataset2Matches$study2_var_id)], NA)
# 
# 
# 
# match1 = 0
# match5 = 0
# match10 = 0
# numberOfMatches = 0
# numberofNA = 0
# varMatchperVar2 = c()
# tc <-as.character(unique(correctVarsData$concept))
# accuracyStats <- list()
# timestamp()
# for(targetConcept in c(tc)){
#     print(targetConcept)
#     match1 = 0
#     match5 = 0
#     match10 = 0
#     numberOfMatches = 0
#     numberofNA = 0
#     correctConceptSubset <- tbl_df(unique(correctVarsData))[unique(correctVarsData)$concept == targetConcept,]  %>% select(concept, dbGaP_studyID_datasetID_varID_1, correctMatches)
#     correctMatchList <- unique(strsplit(unique(correctConceptSubset[,"correctMatches"])[[1]], "; ")[[1]])
#     for(var1 in correctMatchList){
#         dataset1 = gsub('^(.+)\\..+$',"\\1",var1)
#         var1name =  gsub('^.+\\.(.+)$',"\\1",var1)
#         counter = 0
#         for(var2 in setdiff(correctMatchList, var1)){ 
#             dataset2 = gsub('^(.+)\\..+$',"\\1",var2)
#             var2name = gsub('^.+\\.(.+)$',"\\1",var2)
#             dataset2Matches <- textScoreDataDescOnly %>% filter((study1_id == dataset1) & (study1_var_id == var1name) & (studyID_datasetID_2 == dataset2)) %>%
#                 arrange(desc(score)) %>% mutate(rank = dense_rank(desc(score)))
#             var2Rank <- ifelse(var2name %in% dataset2Matches$study2_var_id, dataset2Matches$rank[match(var2name,dataset2Matches$study2_var_id)], NA)
#             #cat('\n',var1, var2, var2Rank, '\n')
#             #print((dataset2Matches %>% select(study1_var_id, study2_var_id, rank)))
#             if(!is.na(var2Rank)){
#                 match1 =  match1 + (var2Rank == 1)
#                 match5 = match5 + (var2Rank <= 5)
#                 match10 = match10 + (var2Rank <= 10)
#                 numberOfMatches = numberOfMatches + 1
#                 counter = counter + 1
#             } else{
#                 #print('NA!!!!!!!!!!')
#                 #cat('\n',var1, var2, var2Rank, '\n')
#                 #print((dataset2Matches %>% select(study1_var_id, study2_var_id, rank)))
#                 numberofNA = numberofNA + 1
#             }
#         }
#         #varMatchperVar2 = c(varMatchperVar2, counter)
#         
#     }
#     #names(varMatchperVar2) <- correctMatchList 
#     accuracyStats <- c(accuracyStats, list(cbind("concept" = targetConcept, "totalMatches" = numberOfMatches + numberofNA, "top1" = match1,  'top5' = match5, "top10" = match10, "NA" =  numberofNA, "Matches(n)" = numberOfMatches)))
#     cat(tc, numberOfMatches + numberofNA, match1, match5, match10, "~ ", numberofNA, numberOfMatches, '\n')
# }
# timestamp()
# 
# ##------ Mon Dec  3 17:04:51 2018 ------##
# #NA means the matched var is not in scores > 0 
# accuracySL <- as.data.frame(do.call(rbind, accuracyStats), stringsAsFactors = F)
# row.names(accuracySL) <- accuracySL$concept
# accuracySL <- accuracySL[order(accuracySL$concept),]
# diffS <- accuracySL[, -c(1)]
# diffS <- sapply(diffS, as.numeric)
# row.names(diffS) <- row.names(accuracySL)
# 
# accuracyDOLS <- as.data.frame(accuracyTotals[order(accuracyTotals$concept),])
# row.names(accuracyDOLS) <- accuracyDOLS$concept
# diffL <- cbind.data.frame("totalMatches" = accuracyDOLS$totalCorrectMatches, 'top1' = accuracyDOLS$top1.Sum, 'top5' = accuracyDOLS$top5.Sum,  "top10 " = accuracyDOLS$top10.Sum,  "NA" = accuracyDOLS$naSum,  "Matches(n)" = accuracyDOLS$matchesCount)
# row.names(diffL) <- row.names(accuracyDOLS)
# 
# diffSL <- diffS - diffL
# row.names(diffSL) <- accuracyDOLS$concept
# 
# accuracyAllLS <- as.data.frame(accuracyTotals.all[order(accuracyTotals.all$concept),])
# row.names(accuracyAllLS) <- accuracyAllLS$concept
# diffL2 <- cbind.data.frame("totalMatches" = accuracyAllLS$totalCorrectMatches, 'top1' = accuracyAllLS$top1.Sum, 'top5' = accuracyAllLS$top5.Sum,  "top10 " = accuracyAllLS$top10.Sum,  "NA" = accuracyAllLS$naSum, "Matches(n)" = accuracyAllLS$matches)
# row.names(diffL2) <- row.names(accuracyAllLS)
# 
# diffSL2 <- diffS-diffL2
# row.names(diffSL) <- accuracyAllLS$concept
# diffLL2 <- diffL-diffL2
# row.names(diffSL) <- accuracyDOLS$concept


#need to create a ranks data frame with desc only and merged data to check accuracy- code below checks accuracy
#tc <- #"Current Smoking Status (smoker in last year)"#"systolic murmur (grade)"#"Current Smoking Status (smoker in last year)"#"Marital Status""Diabetes (binary)" #"systolic murmur (grade)"
# accuracy.t <- rankData %>% dplyr::filter(correctMatchTrue == T) %>% dplyr::mutate(top10 = ifelse(score_rank <= 10, T, F), top5 = ifelse(score_rank <= 5, T, F), top1 = ifelse(score_rank == 1, T, F), totalCorrectMatches = length(unique(strsplit(correctMatches, ";")[[1]]))*(length(unique(strsplit(correctMatches, ";")[[1]]))-1))
# accuracyTotalsByVar.t <- accuracy.t  %>% group_by(dbGaP_studyID_datasetID_varID_1) %>% dplyr::summarize(concept = unique(concept), totalCorrectMatches = unique(totalCorrectMatches), count = n(), NAs = sum(is.na(score_rank)) , top10.count = sum(top10, na.rm = T), top5.count = sum(top5, na.rm = T), top1.count = sum(top1, na.rm = T)) 
# (accuracyTotals <- accuracyTotalsByVar.t %>% group_by(concept) %>% dplyr::summarize(totalCorrectMatches = unique(totalCorrectMatches), matchesCount = sum(count, na.rm = T),  naSum = sum(NAs) + totalCorrectMatches - matchesCount, top10.percent = sum(top10.count, na.rm = T)/sum(count, na.rm = T), top5.percent = sum(top5.count, na.rm = T)/sum(count, na.rm = T), top1.percent = sum(top1.count, na.rm = T)/sum(count, na.rm = T), top1.Sum = sum(top1.count, na.rm = T), top5.Sum = sum(top5.count, na.rm = T), top10.Sum = sum(top10.count, na.rm = T)))
# # %>% mutate(missedMatches = totalCorrectMatches - sumCount)
# 
# accuracy.all <- rankData_all %>% dplyr::filter(correctMatchTrue == T) %>% dplyr::mutate(top10 = ifelse(score_varDescOnly_rank <= 10, T, F), top5 = ifelse(score_varDescOnly_rank <= 5, T, F), top1 = ifelse(score_varDescOnly_rank == 1, T, F), totalCorrectMatches = length(unique(strsplit(correctMatches, ";")[[1]]))*(length(unique(strsplit(correctMatches, ";")[[1]]))-1))
# accuracyTotalsByVar.all <- accuracy.all  %>% group_by(dbGaP_studyID_datasetID_varID_1) %>% dplyr::summarize(concept = unique(concept), totalCorrectMatches = unique(totalCorrectMatches), count = n(), NAs = sum(is.na(score_varDescOnly_rank)) , top10.count = sum(top10, na.rm = T), top5.count = sum(top5, na.rm = T), top1.count = sum(top1, na.rm = T)) 
# (accuracyTotals.all <- accuracyTotalsByVar.all %>% group_by(concept) %>% dplyr::summarize(totalCorrectMatches = unique(totalCorrectMatches), naSum = sum(NAs) + totalCorrectMatches - sum(count),  matches = totalCorrectMatches - naSum, top10.percent = sum(top10.count, na.rm = T)/sum(count, na.rm = T), top5.percent = sum(top5.count, na.rm = T)/sum(count, na.rm = T), top1.percent = sum(top1.count, na.rm = T)/sum(count, na.rm = T), top1.Sum = sum(top1.count, na.rm = T), top5.Sum = sum(top5.count, na.rm = T), top10.Sum = sum(top10.count, na.rm = T)))
# 
# accuracyTotalsSum.all <- accuracyTotals.all %>% dplyr::summarize(totalCorrectMatches = sum(totalCorrectMatches), naSum = sum(naSum),  matches = sum(matches), top10.percent = sum(top10.Sum, na.rm = T)/totalCorrectMatches, top5.percent = sum(top5.Sum, na.rm = T)/totalCorrectMatches, top1.percent = sum(top1.Sum, na.rm = T)/totalCorrectMatches, top1.total = sum(top1.Sum, na.rm = T), top5.total = sum(top5.Sum, na.rm = T), top10.total = sum(top10.Sum, na.rm = T))

# 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#write.csv(heatMapVarMatrixRanksScore, '/Users/laurastevens/Dropbox/Graduate School/Thesis Proposal/ExtractMetaData/Study1VarMatrixbyStudy.csv')


heatMapVarByVarMatrix <- ranksSubset %>% ungroup() %>% filter(dbGaP_studyID_datasetID_varID_2 %in% dbGaP_studyID_datasetID_varID_1) %>%  select(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_varID_2, rank) %>% spread(dbGaP_studyID_datasetID_varID_2, rank)
pheatmap(as.matrix(heatMapVarByVarMatrix))


# textScoreDataDescriptionOnly <- read.csv('/Users/laurastevens/Dropbox/Graduate School/Thesis Proposal/ExtractMetaData/All_Study_Similar_Variables_filtered_description_only_06.24.18.csv', header = T)
# descriptOnlyDatasetsText <- names(table(textScoreDataDescriptionOnly$Study))[2:6]
# descriptOnlyDatasets <- as.character(metaDataVarsInfo[metaDataVarsInfo$Concept == 'Marital Status', c('dataset_name')])
# 
# textScoreDataDescriptionOnly$Study.1 <- gsubfn('\\S+', setNames(as.list(sort(descriptOnlyDatasets)), as.list(sort(descriptOnlyDatasetsText))), as.character(textScoreDataDescriptionOnly$Study.1))
# textScoreDataDescriptionOnly$Study <- gsubfn('\\S+', setNames( as.list(sort(descriptOnlyDatasets)),  as.list(sort(descriptOnlyDatasetsText))), as.character(textScoreDataDescriptionOnly$Study))
# 
# colnames(textScoreDataDescriptionOnly) <- c("study1_bio_id","study1_var_id", "study1_var_nameDBG", "study2_bio_id", "study2_var_id", "study2_var_nameDBG", 'score')
# 
# #Actual Correct Values by Concept
# 
# CorrectVarsData2 <- metaDataVarsInfo[, c(1,6,8)]
# colnames(CorrectVarsData2) <- c("concept", "study1_bio_id", "study1_var_id")
# correctVarsDataConcept2 <- plyr::join(unique(textScoreDataDescriptionOnly[,c("study1_bio_id", 'study1_var_id', 'study1_var_nameDBG', 'score')]), unique(CorrectVarsData2), type = 'inner')
# correctVarsDataConcept2 <- unique(correctVarsDataConcept2)
# correctConceptVars2 <- dplyr::tbl_df(correctVarsDataConcept2) %>% mutate(dbGaP_studyID_datasetID_varID_1 = paste0(study1_bio_id,'.', study1_var_id)) %>% group_by(concept) %>% dplyr::mutate(correctMatches = paste0(dbGaP_studyID_datasetID_varID_1, collapse = '; ')) 
# correctConceptVars2 <- ungroup(correctConceptVars2)
# 
# textScoreDataVettedDatasetsOnly2 <- tbl_df(textScoreDataDescriptionOnly) %>% dplyr::filter(!study1_bio_id == study2_bio_id)
# 
# DataTextScoreConcept2 <- plyr::join(unique(textScoreDataVettedDatasetsOnly2), unique(correctConceptVars2), type = 'inner')
# 
# 
# #filter data to get matches from the datasets that have been vetted
# groupedScoreData_concept2 <- tbl_df(DataTextScoreConcept2) %>% group_by(concept)  %>% 
#     mutate(dbGaP_studyID_datasetID_varID_1 = paste0(study1_bio_id,'.', study1_var_id)) %>% mutate(dbGaP_studyID_datasetID_varID_2 = paste0(study2_bio_id,'.', study2_var_id))  %>%
#     dplyr::filter(!dbGaP_studyID_datasetID_varID_1 == dbGaP_studyID_datasetID_varID_2)
# 
# #check if matched variable is in correct possible matches for each study
# groupedScoreData_concept2 <- groupedScoreData_concept2 %>% group_by(concept,  dbGaP_studyID_datasetID_varID_1,  dbGaP_studyID_datasetID_varID_2) %>%
#     dplyr::mutate(correctMatchTrue = grepl(dbGaP_studyID_datasetID_varID_2, correctMatches))
# 
# #filter to top 10 matches for each variable
# groupedScoreData_concept_study1var_study2dataset2 <- groupedScoreData_concept2 %>% group_by(concept,  dbGaP_studyID_datasetID_varID_1, study2_bio_id) #groups each variable1 (variable 1 = one variable in one study, one file/dataset) and groups those matches by studyDataset2 (study2 by possible datasets in that study) 
# top10matches2 <- dplyr::top_n(groupedScoreData_concept_study1var_study2dataset2, 10, score) 
# 
# #rank top 10 matches and determine if correct matches are in top10, top5, or top (1) match
# top10matchesRanks2 <- top10matches2 %>% dplyr::mutate(rank = c(1:n()))
# accuracy2 <- top10matchesRanks2 %>% dplyr::filter(correctMatchTrue == T) %>% dplyr::mutate(top10 = ifelse(rank < 10, T, F), top5 = ifelse(rank < 5, T, F), top1 = ifelse(rank == 1, T, F))
# 
# #get accuracy for each variable and total accuracy for all mapped variables
# accuracyTotalsByVar2 <- accuracy2 %>% group_by(dbGaP_studyID_datasetID_varID_1) %>% dplyr::summarize(count = n(), top10.count = sum(top10), top5.count = sum(top5), top1.count = sum(top1)) 
# accuracyTotals2 <- accuracyTotalsByVar2 %>% dplyr::summarize(top10.count = sum(top10.count)/sum(count), top5.count = sum(top5.count)/sum(count), top1.count = sum(top1.count)/sum(count)) 
#AccuracyDescOnly <- getAccuracyData(DataTextScoreConcept2)