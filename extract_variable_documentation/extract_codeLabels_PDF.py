import camelot

testFile_pdf = 'C:/Users/pielk/PycharmProjects/automatic-variable-mapping/HF_Action_Data_Dictionary.pdf'

tables = camelot.read_pdf(testFile_pdf, pages='1', flavor='stream')
tables.export('C:/Users/pielk/PycharmProjects/automatic-variable-mapping/foo.csv', f='csv', compress=False)