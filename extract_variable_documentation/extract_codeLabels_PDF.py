import camelot

testFile_pdf = 'HF_Action_Data_Dictionary.pdf'

tables = camelot.read_pdf(testFile_pdf, pages='1', flavor='stream')
tables.export('output/foo.csv', f='csv', compress=False)