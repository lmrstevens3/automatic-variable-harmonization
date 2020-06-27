import pandas as pd
import urllib, urllib2
from tabula import read_pdf

import docx2txt
import camelot

tables = camelot.read_pdf(testFile_pdf)
tables.export('foo.csv', f='csv', compress=True)

testFile_rtf = 'extract_variable_documentation/HF_ACTION_Data_Dictionary.rtf'
testFile_doc = '/Users/laurastevens/Dropbox/Graduate School/BioLINCC files/Framingham cohort/documentation/questionnaires/periodic clinic exam/exam 09/lex0_9.doc'
testFile_pdf = '~/Dropbox/Graduate\ School/BioLINCC\ files/ACCORD/ACCORD Data Dictionary.pdf'

#text = docx2txt.process(testFile_doc)

#text_file = open("/Users/laurastevens/Dropbox/Graduate School/BioLINCC files/aric/Documentation/Databooks/Visit 1/DERIVE13.txt", "r")
#lines = text_file.read()

#testCodes = pd.read_table("/Users/laurastevens/Dropbox/Graduate School/BioLINCC files/aric/Documentation/Databooks/Visit 1/DERIVE13.txt" , sep='\r', skip_blank_lines = True)
#varTest = testCodes.loc[testCodes.columns[0] == 'DRNKR01']

test  = read_pdf(testFile_pdf)