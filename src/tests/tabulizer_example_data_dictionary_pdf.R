#install.packages("tabulizer")
library(tabulizer)
#using HF_ACTION_Data_Dictionary.pdf file as test pdf file using tabulizer package:
    #method = "stream" vs. "decide" (default), yields the same result, method stream is 2 seconds faster
    #method = "lattice" gives an error
pdf_file = "Dropbox/Graduate School/Data Integration and Harmonization/automated_variable_mapping/extract_variable_documentation/HF_Action_Data_Dictionary.pdf"
system.time({ test_tbl1 <- extract_tables(pdf_file, output = "data.frame")})
system.time({ test_tbl2 <- extract_tables(pdf_file, output = "data.frame", method = "stream")})
all.equal(test_tbl1, test_tbl2)
str(test_tbl1)

test_txt1 <- extract_text(pdf_file)
str(test_txt1)

test_meta1 <- extract_metadata(pdf_file)
str(test_meta1)

test_meta1 <- extract_metadata(pdf_file)
str(test_meta1

#for data dictionaires column headers should include all below with Format/Informat missing sometimes 
#Num
#Variable
#Type
#Len
##Format
#Informat 
#Label