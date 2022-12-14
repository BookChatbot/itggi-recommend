import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
from tqdm import tqdm
from db import get_pd_from_table, connect_db, insert_movie_similar, get_not_update_pd_from_table, delete_book_similar, insert_book_similar
import os
from log import info_log, error_log


def get_word2vec_model(books):
    # 불용어 가져오기
    stopwords = []
    with open("data/hangul_stopword.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
            line = line.replace('\n', '')
            stopwords.append(line)

    # summary 전처리
    books['summary'] = books['summary'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    mecab = Mecab()
    corpus_list = []
    for index, row in tqdm(books.iterrows(), total=len(books)):
        text = row['summary']
        tokenized_sentence = mecab.morphs(text)
        stopwords_removed_sentence = [
            word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
        corpus_list.append(stopwords_removed_sentence)
    books['summary'] = corpus_list

    # summary 내용 추가 학습
    word2vec_model = Word2Vec.load('data/ko.bin')
    word2vec_model.wv.save_word2vec_format('data/ko.bin.gz', binary=True)
    word2vec_model = Word2Vec(size=200, window=3, min_count=2, workers=4)
    word2vec_model.build_vocab(corpus_list)
    word2vec_model.intersect_word2vec_format(
        'data/ko.bin.gz', lockf=1.0, binary=True)
    word2vec_model.train(
        corpus_list, total_examples=word2vec_model.corpus_count, epochs=20)

    return word2vec_model


def get_document_vectors(document_list, word2vec_model):
    document_embedding_list = []

    # 각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line:
            if word in word2vec_model.wv.vocab:
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = word2vec_model[word]
                else:
                    doc2vec = doc2vec + word2vec_model[word]

        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)

    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list


def create_new_movies(search_book_id, books_movies):
    idx = books_movies[books_movies['id'] < search_book_id].index
    books_movies = books_movies.drop(idx)
    idx = books_movies[(books_movies['id'] > search_book_id)
                       & (books_movies['id'] <= max_id)].index
    books_movies = books_movies.drop(idx)
    books_movies.reset_index(drop=True, inplace=True)
    return books_movies


def recommendations(books, book_id, cosine_similarities):
    # 책의 제목을 입력하면 해당 제목의 인덱스를 리턴받아 idx에 저장.
    indices = pd.Series(books.index, index=books['id'])
    idx = indices[book_id]

    # 입력된 책과 줄거리(document embedding)가 유사한 책 10개 선정.
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    # 가장 유사한 책 10권의 인덱스
    book_indices = [i[0] for i in sim_scores]

    # 전체 데이터프레임에서 해당 인덱스의 행만 추출. 10개의 행을 가진다.
    recommend = books.iloc[book_indices].reset_index(drop=True)
    return recommend


if __name__ == '__main__':
    # Contents Based 추천 시스템
    try:
        print('books 테이블 불러오는중...')
        DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
        engine, connection, metadata = connect_db(DATABASE_URL)

        books = get_pd_from_table('books', engine, connection, metadata)
        books.drop(['isbn', 'pubDate', 'img', 'rate',
                   'bestseller'], axis=1, inplace=True)
        books.fillna('', inplace=True)
        print(books.head())

        # 책 소개 글에 제목,저자,출판사,장르 추가
        summaries = []
        for book in books.values:
            summary = book[1]
            summary += book[2]
            summary += book[3]
            summary += book[4]
            summary += book[5]
            summaries.append(summary)
        books['summary'] = summaries
        books.drop(['title', 'author', 'publisher',
                   'genre'], axis=1, inplace=True)
        print('books 테이블 전처리 완료')

        word2vec_model = get_word2vec_model(books)
        document_embedding_list = get_document_vectors(
            books['summary'], word2vec_model)
        cosine_similarities = cosine_similarity(
            document_embedding_list, document_embedding_list)
        print('내용 기반 유사도 모델 학습 완료')

        # 유사도가 없는 신간 책만 불러오기
        print('유사도 없는 책들만 불러오기...')
        non_books = get_not_update_pd_from_table(
            'books', 'book_similar', engine, connection, metadata)
        non_books = non_books['id']
        print(type(non_books), len(non_books))

        # books 테이블의 책을 한 권씩 가져와 유사도 높은 책 뽑아내기
        for book_id in non_books:
            delete_book_similar('book_similar', book_id, engine, metadata)
            print(f'{book_id}에 해당하는 기존 추천 책들 삭제 완료')

            rec_books = recommendations(books, book_id, cosine_similarities).id
            rec_books = rec_books.to_list()

            # 추천 받은 책들 book_similar_id 테이블에 업데이트
            for book_similar_id in rec_books:
                insert_book_similar('book_similar', book_id,
                                    book_similar_id, engine, metadata)
            print(f'기준 책: {book_id}, 유사 책: {rec_books}')
        print('추천 책 업데이트를 완료했습니다.')
        info_log('cb 유사도 업데이트 했습니다.')
    except Exception as e:
        print(e)

    # Movie 추천 시스템
    try:
        DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
        engine, connection, metadata = connect_db(DATABASE_URL)

        # 유사도가 없는 신간 책만 불러오기
        print('유사도 없는 책들만 불러오기...')
        books = get_not_update_pd_from_table(
            'books', 'movie_similar', engine, connection, metadata)
        print(books.head(), len(books))

        books.drop(['isbn', 'pubDate', 'img', 'rate',
                   'bestseller'], axis=1, inplace=True)
        books.fillna('', inplace=True)
        print(books.head())

        # 책 소개 글에 제목,저자,출판사,장르 추가
        summaries = []
        for book in books.values:
            summary = book[1]
            summary += book[2]
            summary += book[3]
            summary += book[4]
            summary += book[5]
            summaries.append(summary)
        books['summary'] = summaries
        books.drop(['title', 'author', 'publisher',
                   'genre'], axis=1, inplace=True)
        books.reset_index(drop=True, inplace=True)
        print('books 테이블 전처리 완료')

        print('movies 테이블 불러오는중...')
        movies = get_pd_from_table('movies', engine, connection, metadata)
        movies.drop(['openYear', 'n_code', 'nation', 'runningTime', 'age', 'openDate', 'rate',
                    'participate', 'directors', 'actors', 'blank', 'img', 'genre'], inplace=True, axis=1)
        movies.dropna(axis=0, inplace=True)
        movies.rename(columns={'story': 'summary'}, inplace=True)
        summaries = []
        for movie in movies.values:
            summary = movie[1]
            summary += movie[2]
            summaries.append(summary)
        movies['summary'] = summaries
        movies.reset_index(drop=True, inplace=True)
        print('movies 테이블 전처리 완료')

        max_id = books['id'].max()
        print(f'책id의 최댓값: {max_id}')

        movies_id = [x+max_id for x in movies['id']]
        movies['id'] = movies_id

        books_movies = pd.concat([books, movies], ignore_index=True)
        books_movies.reset_index(drop=True, inplace=True)

        # 새로 만든 books_movies 테이블에 대하여 word2vec 모델 생성
        word2vec_model = get_word2vec_model(books)

        # 책 전체에 대해서 작업 수행
        for search_book_id in books_movies['id']:

            if search_book_id == max_id + 1:
                break

            # 검색할 책 한 권만 포함한 table
            n_books_movies = create_new_movies(search_book_id, books_movies)
            document_embedding_list = get_document_vectors(
                n_books_movies['summary'], word2vec_model)
            cosine_similarities = cosine_similarity(
                document_embedding_list, document_embedding_list)

            rec_movies = recommendations(
                n_books_movies, search_book_id, cosine_similarities).id
            rec_movies = rec_movies.to_list()
            rec_movies = [x - max_id for x in rec_movies]

            print(f'현재 책: {search_book_id} 유사한 영화: {rec_movies}')

            for movie_id in rec_movies:
                insert_movie_similar(
                    'movie_similar', search_book_id, movie_id, engine, metadata)

    except Exception as e:
        error_log(e)
        print(e)
