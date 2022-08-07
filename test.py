import pandas as pd

user_id = str(1)
df = pd.read_csv('data/result_cf.csv', header=0,
                 index_col=[0], usecols=['book_id', user_id], encoding='utf8')
df.sort_values(user_id, ascending=False, inplace=True)

book_id = None
for i, score in enumerate(df[user_id]):
    if 0 < score < 1:
        book_ids = df[i:i+5].index
        break
for book_id in book_ids:
    print(book_id, type(book_id))
