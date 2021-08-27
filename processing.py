import pyodbc
import pandas as pd
import os
import subprocess
import requests
import time
import wget
from multiprocessing import Pool
from pathlib import Path
from tika import parser
from os import listdir, remove
from os.path import isfile, join
import spacy
import en_core_web_sm
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
# from textblob import TextBlob
# import pycountry

# Importing environmental variables library that reads from the .env file
from dotenv import load_dotenv

# Loading key-value pairs from the .env file into the OS environment
load_dotenv()

# Reading the key-value pairs from the OS environment
sql_server = os.getenv('SQL_SERVER')
sql_db = os.getenv('SQL_DATABASE')

conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};' +
                      'Server={};'.format(sql_server) + 
                      'Database={};'.format(sql_db) +
                      'Trusted_Connection=yes;')

##################################################
# Connecting with Database to get DATA IDs
##################################################

with conn:
    cursor = conn.cursor()
    data = pd.read_sql("SELECT * FROM [DS_TEST].[dbo].[app_dash]\
        WHERE (Path LIKE '%Regulatory Document Index-->Facilities-->Gas-->NOVA Gas Transmission Ltd.-->2020 Applications-->2020-10-22 - Application for the NGTL West Path Delivery 2023 Project (GH-002-2020)-->[A-E]%')\
                 AND (SubType = 144) AND Name NOT LIKE 'Receipt%' AND Name NOT LIKE '%Receipt';", conn)
    data.columns = data.columns.str.replace(' ', '')


##################################################
# Ruuning aria2c to Download PDF Files Concurrently
##################################################

# Source: https://www.tutorialexample.com/a-beginner-guide-to-python-use-aria2-to-download-files-on-windows-10/
save_dir = r'G:\app_dash\pdf_aug12'
def get_file_from_cmd(input_file):
    cmd = r'c:\aria2\aria2c.exe -d '+ save_dir +' -i ' + input_file + ' -k 2M -j 9' #Source for parallel download parameters: https://stackoverflow.com/questions/55166245/aria2c-parallel-download-parameters
    try:
        p1=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
        print("---start---")
        msg_content = ''
        for line in p1.stdout:
            print(line)
            l = line.decode(encoding="utf-8", errors="ignore")
            msg_content += l
        p1.wait()
        if '(OK):download completed' in msg_content:
            return True
        return False
    except Exception as e:
        print(e)
        return False
        
# 130 PDFs downloaded in under 90 seconds

##################################################
# Downloading PDF Files of NGTL West path Delivery Project
##################################################

def download_files():
    url_list = []
    for row in data.itertuples():                
        if row.ApplicationLanguage == 'English':
            r = requests.get('https://apps.cer-rec.gc.ca/REGDOCS/File/Download/' + str(row.DataID))
            if 'pdf' in r.headers['content-type']: # To check whther file is pdf or html   
                url_list.append('https://apps.cer-rec.gc.ca/REGDOCS/File/Download/' + str(row.DataID))
    with open('input_file.txt', 'w') as f:
        for url in url_list:
            f.write(url + "\n")
            f.write(' out=' + url.split('/')[-1] + '.pdf' + "\n")
        
    num_lines = len(open('input_file.txt').readlines())
    
    start_time = time.time()
    get_file_from_cmd('input_file.txt')
    duration = round(time.time() - start_time)
    print(f"Downloaded {num_lines/2} files in {duration} seconds ({round(duration / 60, 2)} min or {round(duration / 3600, 2)} hours)")

##################################################
# Extract Text from PDF files and save on database
##################################################

def extract_text():
    pdf_files = [f for f in listdir(save_dir) if isfile(join(save_dir, f))]
    dataIds = [file.split('.')[0] for file in pdf_files]
    content = [parser.from_file("//luxor/data/board/app_dash/pdfs/" + file)["content"] for file in pdf_files]
    df = pd.DataFrame()
    df['dataIds'] = dataIds
    df['content'] = content
    return df
    
def populate_pdfText():
    for row in extract_text().itertuples():
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO dbo.pdfContent(DataID, pdfText) VALUES (?, ?)", (row.dataIds, row.content))
                print(f"text for {row.dataIds} successfully inserted")
        except Exception as e:
            print(f"{row.dataIds}: ERROR! {e}")

    
##################################################
# Validation Work
##################################################

# with conn:
#     cursor = conn.cursor()
#     data = pd.read_sql("SELECT * FROM [DS_TEST].[dbo].[pdfContent] ORDER BY DataID;", conn)
#     for row in data.itertuples():
#         try:
#             print(row.DataID, len(row.pdfText))
#         except Exception as e:
#             print(f"{row.DataID}: ERROR! {e}")
        
#raw = parser.from_file("//luxor/data/board/app_dash/4102482.pdf")
raw = parser.from_file("C:/Users/t1nipun/Desktop/AppDash/4102482.pdf")
text_file = open('4102482.txt',"w",encoding="utf-8")
text_file.write(raw['content'])

##################################################
# Example
##################################################

# with open('example.txt', 'r') as f:
#     text = f.read()
# blob = TextBlob(text)
# print(blob.noun_phrases[:5])
# nlp = en_core_web_sm.load()
# for chunk in nlp(text).noun_chunks:
#     print(chunk.text)

##################################################
# Extract nouns and normalize each word to its root form
##################################################

def normalize_noun_extraction():
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM dbo.pdfContent;", conn)
        
    start_time = time.time()
   
    for row in data.itertuples():
        nlp = en_core_web_sm.load()
        nlp.max_length = 60000000
        try:
            noun_string = " ".join([ent.text for ent in nlp(row.pdfText.lower()) if ent.pos_ == 'NOUN'])
            doc = nlp(noun_string)
            noun_normalize = " ".join([token.lemma_ for token in doc]) # joining all the word tokens after lemmatizer implementation
            remove_single_characters = " ".join( [w for w in noun_normalize.split() if len(w)>1] )
            cursor.execute("UPDATE dbo.pdfContent SET pdfText_Nouns_Lemma = ? WHERE DataID = ?", (remove_single_characters, row.DataID))
            conn.commit()
            print(f"noun extraction and text normalization for {row.DataID} was successful")
        except Exception as e:
            print(f"{row.DataID}: ERROR! {e}")
    duration = round(time.time() - start_time)
    print(f"Noun extraction and lemmatization completed in {duration} seconds ({round(duration / 60, 2)} min or {round(duration / 3600, 2)} hours)")

##################################################
# Extract top keywords from each document using tf-idf scores
##################################################

def top_keywords_tfidf():
    start_time = time.time()
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM [DS_TEST].[dbo].[pdfContent] WHERE pdfText_Nouns_Lemma IS NOT NULL;", conn)
        files = data['DataID'].tolist()
        docs = data['pdfText_Nouns_Lemma'].tolist()
        vectorizer = TfidfVectorizer() #Tfidf vectorizer to calculate tfidf scores
        X = vectorizer.fit_transform(docs)
        files_map = {}
        for idx, value in enumerate(files):
            files_map[idx] = value
        #Create dataframe to store tfidf scores against DataIDs
        feature_names = vectorizer.get_feature_names() 
        corpus_index = [n for n in files_map]
        df = pd.DataFrame(X.T.todense(), index=feature_names, columns=corpus_index)
        df.columns = files

        top_keywords = {}
        for file in files:
            loc = df[file].to_dict()
            loc = {k:v for k,v in loc.items() if v != 0}
            top_tfidf_words = sorted(loc, key=loc.get, reverse=True)[:50]
            top_keywords[file] = top_tfidf_words
        
        for file, keywords in top_keywords.items():
            cursor.execute("UPDATE dbo.pdfContent SET tfidf_top50_keywords = ? WHERE DataID = ?", (", ".join(keywords), file))
    duration = round(time.time() - start_time)
    print(f"Top 50 words by tfidf scores extracted and populated for {len(top_keywords)} files in {duration} seconds ({round(duration / 60, 2)} min or {round(duration / 3600, 2)} hours)")
    
##################################################
# Extract noun phrases
##################################################

def extract_noun_phrases():
    start_time = time.time()
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM dbo.pdfContent;", conn)
        
        
        dict_noun_phrase = {}
        for row in data[:1].itertuples():
            blob = TextBlob(row.pdfText)
            dict_noun_phrase[row.DataID] = list(set(blob.noun_phrases))
            print(len(dict_noun_phrase[row.DataID]))
        
        for file, noun_phrases in dict_noun_phrase.items():
            for noun_phrase in noun_phrases:
                cursor.execute("INSERT INTO dbo.nounPhrases(DataID, nounPhrase) VALUES (?, ?)", (file, noun_phrase))
    duration = round(time.time() - start_time)
    print(f"Noun phrase extraction for {len(dict_noun_phrase)} file(s) completed in {duration} seconds ({round(duration / 60, 2)} min or {round(duration / 3600, 2)} hours)")

##################################################
# Extract Submitter Type
##################################################

def extract_submitter_type():
    start_time = time.time()
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM [DS_TEST].[dbo].[app_dash]\
        WHERE (Path LIKE '%Regulatory Document Index-->Facilities-->Gas-->NOVA Gas Transmission Ltd.-->2020 Applications-->2020-10-22 - Application for the NGTL West Path Delivery 2023 Project (GH-002-2020)-->[A-E]%')\
                 AND (SubType = 144) AND Name NOT LIKE 'Receipt%' AND Name NOT LIKE '%Receipt';", conn)
        files = []
        submitter_type = []
        for row in data.itertuples():
            if "C - NOVA" in row.Path:
                files.append(row.DataID)
                submitter_type.append('Applicant')
            elif "A - Commission" in row.Path:
                files.append(row.DataID)
                submitter_type.append('Commission')
            elif "B - Canada" in row.Path:
                files.append(row.DataID)
                submitter_type.append('CER')
            elif "D - Intervenors" in row.Path:
                files.append(row.DataID)
                submitter_type.append('Intervenor')
            else:
                files.append(row.DataID)
                submitter_type.append('Commenter')
        df = pd.DataFrame(list(zip(files, submitter_type)),
               columns =['DataID', 'SubmitterType'])
        
        for df_row in df.itertuples():
            cursor.execute("UPDATE dbo.pdfContent SET submitterType = ? WHERE DataID = ?", (df_row.SubmitterType, df_row.DataID))
    duration = round(time.time() - start_time)
    print(f"Submitter type extracted and populated in {duration} seconds ({round(duration / 60, 2)} min or {round(duration / 3600, 2)} hours)")
    
##################################################
# Word Frequency
##################################################
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

def word_frequency():
    start_time = time.time()
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM dbo.pdfContent WHERE pdfText_Nouns_Lemma IS NOT NULL AND submitterType = 'Intervenor';", conn)
        print(len(data))
        str = ''
        for row in data.itertuples():
            str += row.pdfText_Nouns_Lemma + ' '
        wordlist = str.split()
        dictionary = wordListToFreqDict(wordlist)
        sort_dict = sortFreqDict(dictionary)
        #df = pd.DataFrame.from_dict(sort_dict)
    #return df.to_csv('experiment.csv')
    print(sort_dict)

##################################################
# TESTING Seqential Code
##################################################

def extract_text():
    bad_files = []
    start_time = time.time()
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM dbo.pdfContent;", conn)
        for row in data.itertuples():
            response = requests.get('https://apps.cer-rec.gc.ca/REGDOCS/File/Download/' + str(row.DataID))
            results = parser.from_buffer(response)
            if results['content'] is None:
                bad_files.append('https://apps.cer-rec.gc.ca/REGDOCS/File/Download/' + str(row.DataID))
            else:
                cursor.execute("UPDATE dbo.pdfContent SET sampleText = ? WHERE DataID = ?", (results['content'], row.DataID))      
    duration = round(time.time() - start_time)
    with open('bad_files.txt', 'w') as f:
        for file in bad_files:
            f.write(file + "\n")
    print(f"Text extracted and populated in database from {len(data)-len(bad_files)} PDF file(s) in {duration} seconds ({round(duration / 60, 2)} min or {round(duration / 3600, 2)} hours)")


def get_DataID():
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM dbo.pdfContent;", conn)
        ids = [id for id in data['DataID']]
    return ids

def extract_text2(DataID):
    bad_files = []
    response = requests.get('https://apps.cer-rec.gc.ca/REGDOCS/File/Download/' + str(DataID))
    results = parser.from_buffer(response)
    if results['content'] is None:
        bad_files.append('https://apps.cer-rec.gc.ca/REGDOCS/File/Download/' + str(DataID))
    else:
        with conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE dbo.pdfContent SET sampleText1 = ? WHERE DataID = ?", (results['content'], DataID))      
    return bad_files

def process_handler():
    start_time = time.time()

    pdf_ids = get_DataID()
    with Pool() as pool:
        results = pool.map(extract_text2, pdf_ids, chunksize=1)
    with open('bad_files1.txt', 'w') as f:
        for result in results:
            if len(result) != 0:
                f.write(result[0] + "\n")
    
    duration = round(time.time() - start_time)
    print(f"Text extracted and populated in {duration} seconds ({round(duration / 60, 2)} min or {round(duration / 3600, 2)} hours)")

def char_len():
    with conn:
        cursor = conn.cursor()
        data = pd.read_sql("SELECT * FROM dbo.pdfContent WHERE pdfText IS NOT NULL;", conn)
        df = pd.DataFrame()
        df['dataIds'] = data['DataID']
        df['pdfText'] = data['pdfText'].apply(len)
        df['sampleText'] = data['sampleText'].apply(len)
        df['sampleText1'] = data['sampleText1'].apply(len)
    return df.to_csv('test.csv', encoding = 'utf-8-sig')


            
        
if __name__ == "__main__":
    #download_files()
    # populate_pdfText()
    # normalize_noun_extraction()
    # top_keywords_tfidf()
    # extract_noun_phrases()
    # extract_submitter_type()
    # word_frequency()
    #extract_text1()
    #process_handler()
    char_len()