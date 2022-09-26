import pandas as pd
import os
from log import info_log, error_log
import h5py
from konlpy.tag import Mecab
from tqdm import tqdm
from gensim.models import Word2Vec
from db import get_pd_from_table, connect_db
from sklearn.metrics.pairwise import cosine_similarity


DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
engine, connection, metadata = connect_db(DATABASE_URL)


#책 데이터 가져오는 함수
def get_book_data():
    books = get_pd_from_table('books', engine, connection, metadata)
    books.drop(['Unnamed: 0', 'sense'], inplace=True, axis=1)
    books.fillna('', inplace=True)
    summaries = []
    for book in books.values:
        summary = book[1]
        summary += book[2]
        summary += book[3]
        summary += book[4]
        summary += book[5]
        summaries.append(summary)
    books['summary'] = summaries
    books.drop(['author', 'publisher', 'genre'], axis=1, inplace=True)
    books.reset_index(drop=True, inplace=True)

    return books


#영화 데이터 가져오는 함수
def get_movie_data():
    movies = get_pd_from_table('movies', engine, connection, metadata)
    movies.drop(['Unnamed: 0', 'openYear', 'n_code', 'nation', 'runningTime', 'age', 'openDate', 'rate', 'participate', 'directors', 'actors', 'blank', 'img', 'genre'], inplace=True, axis=1)
    movies.dropna(axis=0, inplace=True)
    movies.rename(columns={'story':'summary'}, inplace=True)
    summaries = []
    for movie in movies.values:
        summary = movie[1]
        summary += movie[2]
        summaries.append(summary)
    movies['summary'] = summaries
    movies.reset_index(drop=True, inplace=True)

    return movies


#책, 영화 데이터 합치기
def concat_books_moives(books, movies):
    books = get_pd_from_table('books', engine, connection, metadata)
    movies = pd.read_csv('movies', engine, connection, metadata)
    books_movies = pd.concat([books, movies], ignore_index=True)
    books_movies.reset_index(drop=True, inplace=True)

    return books_movies


def first_movie_id():
    movies = pd.read_csv('movies', engine, connection, metadata)
    books = get_pd_from_table('books', engine, connection, metadata)
    max_id = books['id'].max()
    movies_id = [x+max_id for x in movies['id']]
    movies['id'] = movies_id

    return movies_id


def get_cosine_similarity(books_movies):
    # 불용어 가져오기
    stopwords = []
    with open("data/hangul_stopword.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
            line = line.replace('\n', '')
            stopwords.append(line)

    # summary 전처리
    books_movies['summary'] = books_movies['summary'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    mecab = Mecab()
    corpus_list = []
    for index, row in tqdm(books_movies.iterrows(), total=len(books_movies)):
        text = row['summary']
        tokenized_sentence = mecab.morphs(text)
        stopwords_removed_sentence = [
            word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
        corpus_list.append(stopwords_removed_sentence)
    books_movies['summary'] = corpus_list

    # summary 내용 추가 학습
    word2vec_model = Word2Vec.load('data/ko.bin')
    print(f'현재 모델의 단어 개수: {len(word2vec_model.wv.vocab)}')
    word2vec_model.wv.save_word2vec_format('data/ko.bin.gz', binary=True)
    word2vec_model = Word2Vec(size=200, window=3, min_count=2, workers=4)

    word2vec_model.build_vocab(corpus_list)
    word2vec_model.intersect_word2vec_format(
        'data/ko.bin.gz', lockf=1.0, binary=True)
    word2vec_model.train(
        corpus_list, total_examples=word2vec_model.corpus_count, epochs=20)


