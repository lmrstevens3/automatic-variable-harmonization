#library(plyr)
library(dplyr)
library(tidyr)
library(DT)
library(plotly)
library(data.table)
library(parallel)
library(multidplyr)
library(processx)



getRankData <- function(data, variable){
    data$score <- data[[variable]] #set the current score column so can refer to it when doing manipulations
    #groups each variable1 (variable 1 = one variable in one study  file/dataset) and groups those matches by studyDataset2 (study2 by possible datasets in that study)
    rankData <- data %>% group_by(concept,  dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% arrange(desc(score), .by_group = T)  %>% 
        mutate(rank = dense_rank(desc(score))) %>% mutate(rank = replace(rank, score == 0, NA)) %>%
        mutate(top10predMatches = length(which(rank <= 10)), top5predMatches = length(which(rank <= 5)), topPredMatches = length(which(rank <= 1)))
    #makes sure rank column name is for the correct score and order columns 
    rankVars <- c("rank", "top10predMatches", 'top5predMatches', "topPredMatches")
    names(rankVars) <- paste0(rankVars, "_", variable)
    rankData <- rankData %>% dplyr::select(variable, rankVars, everything()) %>% dplyr::select(-score) 
    rankData <- rankData[,c(6:length(rankData),1:5)]
    return(rankData)
}


addRankColsForEachScoreFunction <- function(data, score_variables){
    for(i in score_variables){
        paste0("calculating ", i, " ranks")
        data <- getRankData(data, i)
    }
    return(data)
}

refVarAccuracyFunction <- function(accuracyData, scoreVariable){
    rankVars <- grep(paste0("_",scoreVariable,"$"), colnames(accuracyData), value =T)
    accuracyData[, c("rank", "top10pred", "top5pred", "top1pred")] <- accuracyData[,rankVars]
    #determine if top 1,5,10 ranked vars has correct match 
    accuracyData <- accuracyData %>% ungroup() %>% dplyr::mutate(top10 = ifelse(rank <= 10, T, F), top5 = ifelse(rank <= 5, T, F), top1 = ifelse(rank == 1, T, F))
    #get var accuracy
    accuracyTotalsByVar <- accuracyData %>% group_by(dbGaP_studyID_datasetID_varID_1, dbGaP_studyID_datasetID_2) %>% mutate(numVarsInMatchedDataset = length(dbGaP_studyID_datasetID_varID_2)) %>%
        group_by(dbGaP_studyID_datasetID_varID_1) %>% 
        dplyr::summarize(concept = unique(concept), dbGaP_studyID_datasetID_1 = unique(dbGaP_studyID_datasetID_1),
                         totalPossibleMatches = unique(totalPossibleMatches), totalMatches = length(which(!is.na(rank))), totalVarsInConcept = unique(totalVarsInConcept), 
                         top10.TP = sum(top10, na.rm = T), top10.TP.alt = sum(top10/numConceptVarsInDataset), top10.pred = sum(top10pred/numVarsInMatchedDataset),
                         top5.TP = sum(top5, na.rm = T), top5.TP.alt = sum(top5/numConceptVarsInDataset), top5.pred = sum(top5pred/numVarsInMatchedDataset), 
                         top1.TP = sum(top1, na.rm = T), top1.TP.alt = sum(top1/numConceptVarsInDataset), top1.pred  = sum(top1pred/numVarsInMatchedDataset)) 
    return(accuracyTotalsByVar)
}

accuracyTotalsFunction <- function(accuracyData, scoreVariable){
    #get accuracy by reference variable
    accuracyTotalsByVar <- refVarAccuracyFunction(accuracyData, scoreVariable)
    #get accuracy by concept
    accuracyTotalsByConcept <- accuracyTotalsByVar %>% group_by(concept) %>% 
        dplyr::summarize(totalPossibleMatches = unique(totalPossibleMatches),  totalVarsInConcept = unique(totalVarsInConcept), totalDatasetsInConcept = length(unique(dbGaP_studyID_datasetID_1)), totalPossibleMatches.alt = totalVarsInConcept*(totalDatasetsInConcept-1), totalMatches = sum(totalMatches),
                         top10.TP = sum(top10.TP, na.rm = T), top10.TP.alt = sum(top10.TP.alt), top10.predictedMatches = sum(top10.pred), top10.precision = top10.TP/top10.predictedMatches, top10.recall = top10.TP/totalPossibleMatches, top10.recall.alt = top10.TP.alt/(totalVarsInConcept*(totalDatasetsInConcept-1)), top10.F1 = 2*top10.precision*top10.recall/(top10.precision+top10.recall), top10.F1.alt = 2*top10.recall.alt*top10.recall.alt/(top10.recall.alt+top10.recall.alt),
                         top5.TP = sum(top5.TP, na.rm = T), top5.TP.alt = sum(top5.TP.alt), top5.predictedMatches = sum(top5.pred), top5.precision = top5.TP/top5.predictedMatches, top5.recall = top5.TP/totalPossibleMatches, top5.recall.alt = top5.TP.alt/(totalVarsInConcept*(totalDatasetsInConcept-1)), top5.F1 = 2*top5.precision*top5.recall/(top5.precision+top5.recall), top5.F1.alt = 2*top5.recall.alt*top5.recall.alt/(top5.recall.alt+top5.recall.alt),
                         top1.TP = sum(top1.TP, na.rm = T), top1.TP.alt = sum(top1.TP.alt), top1.predictedMatches  = sum(top1.pred), top1.precision = top1.TP/top1.predictedMatches, top1.recall = top1.TP/totalPossibleMatches, top1.recall.alt = top1.TP.alt/(totalVarsInConcept*(totalDatasetsInConcept-1)), top1.F1 = 2*top1.precision*top1.recall/(top1.precision+top1.recall), top1.F1.alt = 2*top1.recall.alt*top1.recall.alt/(top1.recall.alt+top1.recall.alt)) %>%
        mutate_at(vars(matches(".F1")), funs(replace(.,is.na(.),0)))
    #get total accuracy
    accuracyTotals <- accuracyTotalsByConcept  %>% 
        dplyr::summarize(totalPossibleMatches = sum(totalPossibleMatches), totalVarsMatched = sum(totalVarsInConcept), totalPossibleMatches.alt = sum(totalPossibleMatches.alt), totalMatches = sum(totalMatches),
                         top10.TP = sum(top10.TP, na.rm = T), top10.TP.alt = sum(top10.TP.alt), top10.predictedMatches = sum(top10.predictedMatches), top10.precision = top10.TP/top10.predictedMatches, top10.recall = top10.TP/totalPossibleMatches, top10.recall.alt = top10.TP.alt/totalPossibleMatches.alt, top10.F1 = 2*top10.precision*top10.recall/(top10.precision+top10.recall), top10.F1.alt = 2*top10.recall.alt*top10.recall.alt/(top10.recall.alt+top10.recall.alt),
                         top5.TP = sum(top5.TP, na.rm = T), top5.TP.alt = sum(top5.TP.alt), top5.predictedMatches = sum(top5.predictedMatches), top5.precision = top5.TP/top5.predictedMatches, top5.recall = top5.TP/totalPossibleMatches, top5.recall.alt = top5.TP.alt/totalPossibleMatches.alt, top5.F1 = 2*top5.precision*top5.recall/(top5.precision+top5.recall), top5.F1.alt = 2*top5.recall.alt*top5.recall.alt/(top5.recall.alt+top5.recall.alt),
                         top1.TP = sum(top1.TP, na.rm = T), top1.TP.alt = sum(top1.TP.alt), top1.predictedMatches  = sum(top1.predictedMatches), top1.precision = top1.TP/top1.predictedMatches, top1.recall = top1.TP/totalPossibleMatches, top1.recall.alt = top1.TP.alt/totalPossibleMatches.alt, top1.F1 = 2*top1.precision*top1.recall/(top1.precision+top1.recall), top1.F1.alt = 2*top1.recall.alt*top1.recall.alt/(top1.recall.alt+top1.recall.alt)) %>% 
        mutate(concept = "All Concepts") %>% mutate_at(vars(matches(".F1")), funs(replace(.,is.na(.),0)))
    accuracyResults <- accuracyTotalsByConcept %>% bind_rows(accuracyTotals)
    return(accuracyResults)
}


#get accuracy differences
differencesFunction <- function(data, variablesToCompare){
    combos <- as.data.frame(combn(variablesToCompare,2))[1:length(variablesToCompare)-1]
    print(combos)
    for(i in 1:length(combos)){
        var1 <- as.character(combos[1,i])
        var2 <- as.character(combos[2,i])
        data <- eval(substitute(mutate(data, diff = var2 - var1), 
                                    list(var1 = as.name(var1), var2 = as.name(var2))))
        colnames(data)[which(colnames(data) %in% "diff")] <- paste0(var1,".DIFF_FROM.", var2)
    }
    return(data)
}


#################################################################################
################ Accuracy Main Script
scoresFile <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_similarity_scores_and_combined_scores_ManuallyMappedConceptVars_2.17.20.csv'
manualConceptVarMapFile = '~/Dropbox/Graduate School/Data Integration and Harmonization/Manual Concept Variable Mappings BioLINCC and DbGaP/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv'
rankFileOut <- '~/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/FHS_CHS_MESA_ARIC_all_scores_ranked_ManuallyMappedConceptVars_3.17.20.csv'
accuracyXlsxFileOut = '/Users/laurastevens/Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/accuracy_plusalternate_VarMappingsBagOfWordsAllScores_Mar2020_V2.xlsx'



#!!note!!- important to make sure na.strings and quotes are read in the same for files to merge correctly- files are written with NA values as "" for interoperability, and some strings have quotes
allScoresData <- fread(scoresFile , header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))
#mannually mapped concepts and possible matches
conceptMappedVarsData <- fread(manualConceptVarMapFile, header = T, sep = ',', na.strings = "", stringsAsFactors=FALSE, showProgress=getOption("datatable.showProgress", interactive()))
tbl_df(conceptMappedVarsData) %>% dplyr::select(conceptID, study_1, dbGaP_dataset_label_1, varID_1, var_desc_1, totalVarsInConcept, ends_with("Dataset"), ends_with("Matches"))

#Join scores data and concept data and filter out datasets not in goldstandard and concepts with broad definitions that don't map across variables/studies (ex. cholesterol lowering med because not same medication for mappings)
#takes about 3-4 min with 21267777 X 41 and 1703 X 21
rankData2 <- allScoresData %>% dplyr::left_join((conceptMappedVarsData %>% select(-data_desc_1, -detailedTimeIntervalDbGaP_1, -var_coding_labels_1))) %>% 
    filter(!concept %in% c("Cholesterol Lowering Medication")) %>% filter(dbGaP_studyID_datasetID_2 %in% dbGaP_studyID_datasetID_1)
#, "unique Participant Identifier"
#check join worked correctly and remove all scores data so its not taking up memory
unique(rankData2$concept) #52 concepts, 2.99 GB, 6754451 X 47 after merge and filter (20485083 X 47, 4.5GB if use PID concept)
object.size(rankData)
rm(allScoresData)

#set up parallel processing so things run faster
cores <- detectCores() - 2 # mac can split 4 cores into 8 cores for parallel procesing
cluster <- makeCluster(cores)
set_default_cluster(cluster)
#mark  which matches are true/false, make a ref/matched variable for display- takes about 10 min for 6754451 X 47
timestamp()
rankData2 <-  partition(rankData2, dbGaP_studyID_datasetID_varID_1) %>% group_by(dbGaP_studyID_datasetID_varID_2) %>% 
    dplyr::mutate(correctMatchTrue = grepl(paste0('(^|; )', dbGaP_studyID_datasetID_varID_2, '(; |$)'), possibleMatches)) %>% 
    dplyr::mutate(reference_variable = paste0(unique(na.omit(c(study_1, dbGaP_dataset_label_1, var_desc_1))), collapse = "."), 
                  matched_variable = paste0(unique(na.omit(c(study_2, dbGaP_dataset_label_2, var_desc_2))), collapse = ".")) %>%
    collect() %>% ungroup() %>% 
    dplyr::select(dbGaP_studyID_datasetID_varID_1,  dbGaP_studyID_datasetID_varID_2, correctMatchTrue, conceptID,  reference_variable, matched_variable, concept, study_1, study_2, varID_1, varID_2, units_1, units_2, var_coding_counts_distribution_1, var_coding_counts_distribution_2, timeIntervalDbGaP_1, timeIntervalDbGaP_2, dbGaP_studyID_datasetID_1, dbGaP_studyID_datasetID_2, metadataID_1, metadataID_2, matchID, concept, numConceptVarsInDataset, matchesInSameDataset, totalVarsInConcept, totalPossibleMatches, starts_with("score_"))
timestamp()
on.exit(stopCluster(cluster))

#bash command 
#FOO is assigned to ConceptMappedVarsFile: FOO='manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
#grep '"CHD"' $FOO > BAR
#perl -pe 's/,,/,\"NA\",/g' BAR | perl -pe 's/,,/,\"NA\",/g' | perl -pe 's/,(\d+),/,\"$1\",/g' | perl -pe 's/,(\d+),/,\"$1\",/g' | perl -pe 's/\",\"/\t/g' | cut -f 13,14,19 | column -t
#replace commas inbetween quotes with nothing
#perl -pe 's/"(.+?[^\\])"/($ret = $1) =~ (s#,##g); $ret/ge' SHAW | perl -pe 's/,/\t/g'
### perl -pe 's/"(.+?[^\\])"/($ret = $1) =~ (s#,##g); $ret/ge' input.foo | perl -pe 's/\s*,\s*/\t/g'
#admins-MacBook-Pro-2:automated_variable_mapping laurastevens$ perl -pe 's/"(.+?[^\\])"/($ret = $1) =~ (s#,##g); $ret/ge' SHAW | perl -pe 's/,/\t/g' | cut -f 1-4,29
#admins-MacBook-Pro-2:automated_variable_mapping laurastevens$ head -n1 $FOO > BAR
#admins-MacBook-Pro-2:automated_variable_mapping laurastevens$ grep ",CHD," $FOO >> BAR
#head -n1 $FOO > BAR
#grep ",CHD," $FOO >> BAR
#perl -pe 's/"(.+?[^\\])"/($ret = $1) =~ (s#,##g); $ret/ge' BAR | perl -pe 's/,/\t/g' | cut -f 1-4,29 | awk '($4 == "CHD")'| awk '$5 == 1'| column -t | wc -l
#get ranks for all scores
possibleScores <- grep("score_",colnames(rankData)[-c(which(colnames(rankData)  %in% c("score_euclidianDistance")))], value = T)
timestamp()#takes about 10 min with 6754451
rankData  <- addRankColsForEachScoreFunction(rankData, possibleScores)
timestamp()
fwrite(rankData, rankFileOut, sep = ',', qmethod = 'double', na = "", row.names = F)

#get precision, recall, and F1
accuracyData <- rankData %>% ungroup() %>% dplyr::filter(correctMatchTrue == T)
accuracyTotals <- sapply(possibleScores, function(x){print(paste0("calculating ", x, " accuracy")); accuracyTotalsFunction(accuracyData, x)}, USE.NAMES = T, simplify = F)
accuracyTotals[["overallAccuracyAllScores"]] <- do.call(rbind, accuracyTotals) %>% filter(concept == "All Concepts") %>% dplyr::mutate(score = names(accuracyTotals)) %>% 
    dplyr::select(score, totalVarsMatched, totalPossibleMatches, totalPossibleMatches.alt, totalMatches, ends_with(".TP"), ends_with(".TP.alt"), ends_with(".precision"), ends_with(".recall"), ends_with("recall.alt"), ends_with("F1"), ends_with("F1.alt"))

#write results to a file
write.xlsx(as.data.frame(accuracyTotals[[length(accuracyTotals)]]), accuracyXlsxFileOut, sheetName = names(accuracyTotals[length(accuracyTotals)]), row.names = F)
sapply(names(accuracyTotals)[-length(accuracyTotals)], function(accuracyName){
    write.xlsx2(x=as.data.frame(accuracyTotals[[accuracyName]]), file=accuracyXlsxFileOut, sheetName = gsub("score_","",accuracyName), row.names = F, append = T)
})

#get differences in accuracy by concept and plot 
accuracyDifferencesData <- lapply(accuracyTotals[c(1,11,13,15)], function(x){returnData <- as.data.frame(x %>% dplyr::select(top1.precision, top1.recall, top1.F1)); rownames(returnData) <- x$concept; return(returnData)})
accuracyDifferencesData <- do.call('cbind', accuracyDifferencesData) 
accuracyDifferences.top1recall <- differencesFunction(accuracyDifferencesData %>% select(ends_with(".recall")), grep(".recall", colnames(accuracyDifferencesData), value = T))
rownames(accuracyDifferences.top1recall) <- rownames(accuracyDifferencesData)

plot_ly(x = accuracyDifferences.top1recall$score_desc.top1.recall.DIFF_FROM.score_desc_SqNoisyOr_maxOtherMeta_euclidean.top1.recall*(100), y = row.names(accuracyDifferences.top1recall), type = 'bar') %>% 
    layout(xaxis = list(title = "Recall Difference"), title ="variables scored by descriptions vs. combined with max score from variables scored on units/codelabels/distribtuion", font = list(size = 8))
plot_ly(x = accuracyDifferences.top1recall$score_desc.top1.recall.DIFF_FROM.score_descUnits_SqNoisyOr_codeLabEuclidean.top1.recall*(100), y = row.names(accuracyDifferences.top1recall), type = 'bar') %>% 
    layout(xaxis = list(title = "Recall Difference"), title ="variables scored by descriptions vs. descriptions+units scores combined with codelabels/distribtuion scores", font = list(size = 8))
plot_ly(x = accuracyDifferences.top1recall$score_desc.top1.recall.DIFF_FROM.score_desc_SqNoisyOr_codeLabEuclidean.top1.recall*(100), y = row.names(accuracyDifferences.top1recall), type = 'bar') %>% 
    layout(xaxis = list(title = "Recall Difference"), title ="variables scored by descriptions vs. descriptions scores combined with codelabels/distribtuion scores", font = list(size = 8))

