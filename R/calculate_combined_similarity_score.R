library(dplyr)
library(data.table)
library(docstring)

#Laura Stevens
#project directory: ~/Dropbox/Graduate School/Data Integration and Harmonization/


calc_noisyOR <- function(data, parent_col, child_cols){
    #' Calculates noisy or for parent and child columns a data frame. 
    #' If 2 child columns provided noisy or is: 1-((1-parent)^(1+child1^2+child2^3)). If 3 child columns are provided 2 and 3rd child are both cubed. 
    #' 
    #' @param data: a dataframe containing parent and child columns  
    #' @param parent_col: a column in data that contains ref ID's.
    #' @param child_cols: a list of columns in data to use for the child cols (orderd by which child should provide the most influence on the final score).
    exp <- (1+data[[child_cols[1]]]^2+data[[child_cols[2]]]^3)
    if(length(child_cols) == 3){
        exp <- (1+data[[child_cols[1]]]^2+data[[child_cols[2]]]^3+data[[child_cols[3]]]^3)
    }
    return(1-(1-data[[parent_col]])^(exp))
}








