import pandas as pd
from dataprep_lite.preprocessing import StopWordRemover, TfidfVectorizerWrapper

if __name__ == '__main__':
    df = pd.DataFrame({'text': ['This is a sample sentence', 'Another example text']})
    remover = StopWordRemover(columns_to_process=['text'])
    df_clean = remover.fit_transform(df)
    tfidf = TfidfVectorizerWrapper(columns_to_process=['text'])
    df_vec = tfidf.fit_transform(df_clean)
    print(df_vec)
