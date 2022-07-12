import streamlit as st
import datasketch as ds
import pandas as pd
import re
from stop_words import get_stop_words



def cut_word(s):
    """
    Cut words
    """
    # stop words
    stop_words = get_stop_words('english')
    # delete url
    s = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(s))
    # Remove punctuation from text
    s = re.sub("\W", " ", s.lower())
    # cut words
    s = s.split(' ')
    outstr = ''
    # remove stop words
    for word in s:
        if word not in stop_words and len(word)>1:
            outstr += word
            outstr += " "
    return  outstr[:-1]  

def split_num(s, num=2):
    """
    Divide by num
    """
    s = s.split(' ')
    res = []
    for i in range(len(s)):
        res.append(' '.join(s[i:i+num]))
    return res


def compute_hash(one, two, num_perm=100):
    """
    Calculate hash
    """
    hash1 = ds.MinHash(num_perm=num_perm)
    hash2 = ds.MinHash(num_perm=num_perm)
    for i in one:
        hash1.update(i.encode("utf8"))
    for i in two:
        hash2.update(i.encode("utf8"))
    return hash1.jaccard(hash2)

df = pd.read_csv('train.csv')
df = df[:1000]
df['cut word result'] = df['text'].apply(cut_word)

st.header("Results")
st.header("Analysis Process")
st.markdown("""
- Read data
- Sentence splitting and deactivation
- characteristic matrix
- Calculate LSH
- Calculate Jaccard similarity""")

st.subheader("Cut Word Result")
st.dataframe(df)


st.subheader("Characteristic matrix")
shingle_size = st.text_input('Shingle size', '2')
df['characteristic matrix'] = df['cut word result'].apply(split_num, num=int(shingle_size))
st.dataframe(df['characteristic matrix'])


st.subheader("Select text")
num_perm = st.text_input('Number of random permutation functions', 128)
text1 = st.selectbox("Select bag 1:", list(df['characteristic matrix']), index=0)
text2 = st.selectbox("Select bag 2:", list(df['characteristic matrix']), index=1)

st.write("Estimated Jaccard for text1 and text2 is: ", compute_hash(text1, text2, int(num_perm)))
s1 = set(text1)
s2 = set(text2)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
st.write("Actual Jaccard for text1 and text2 is: ", actual_jaccard)


st.write('text1 with other text values')
save_hash = []
save_actual_jaccard = []
for i in range(len(df)):
    if df['characteristic matrix'][i]!=text1:
        save_hash.append(compute_hash(text1, df['characteristic matrix'][i], int(num_perm)))
        
        s1 = set(text1)
        s2 = set(df['characteristic matrix'][i])
        actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
        save_actual_jaccard.append(actual_jaccard)
chart_data = pd.DataFrame()
chart_data['Estimated Jaccard'] = save_hash
chart_data['Actual Jaccard'] = save_actual_jaccard
st.line_chart(chart_data)

st.write('\n\n\n')
st.subheader("Code URL")
st.markdown("""From: [Github](https://github.com/tina5980g/Data-Science-1-Project-T2)""")