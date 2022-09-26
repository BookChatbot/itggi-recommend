from cf import get_similar_by_cf, predict_rating
from log import info_log, error_log
import pandas as pd
from db import delete_user_similar, get_pd_from_table, insert_user_similar, insert_movie_similar, get_list_from_table, delete_book_similar, insert_book_similar, connect_db, get_not_update_pd_from_table
import h5py
import os
from movie_cb import get_book_data, get_movie_data, concat_books_moives, rearrange_movie_id, create_new_movies

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
engine, connection, metadata = connect_db(DATABASE_URL)

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


#Contents Based 추천 시스템
try:
    #책 전체 불러오기
    print('books 테이블 불러오는 중...')
    books = get_book_data()
    print(books.head())

    #영화 불러오기
    print('books 테이블 불러오는 중...')
    movies = get_movie_data()
    print(movies.head())

    #책, 영화 데이터 합치기
    print('books, movie 테이블 합치는 중...')
    books_movies = concat_books_moives()
    print(books_movies.head())

    #영화 인덱스 재설정
    max_id, movies_id = rearrange_movie_id()

    # 유사도가 없는 영화만 불러오기
    print('유사도 없는 영화들만 불러오기...')
    non_movies = get_not_update_pd_from_table(
        'movies', 'movie_similar', engine, connection, metadata)
    non_movies = non_movies['id']
    print(type(non_movies), len(non_movies))

    # load numpy array from h5 file
    h5f = h5py.File('data/result_cb.h5', 'r')
    cosine_similarities = h5f['similarity'][:]
    h5f.close()
    print('저장된 유사도 모델 h5 불러오기 완료')

    # # books 테이블의 책을 한 권씩 가져와 유사도 높은 책(영화) 뽑아내기
    # for book_id in non_movies:
    #     delete_book_similar('movie_similar', book_id, engine, metadata)
    #     print(f'{book_id}에 해당하는 기존 추천 책들 삭제 완료')

    #     rec_books = recommendations(books_movies, book_id, cosine_similarities).id
    #     rec_books = rec_books.to_list()

    #     # 추천 받은 책들 book_similar_id(movie_similar_id) 테이블에 업데이트
    #     for movie_similar_id in rec_books:
    #         insert_movie_similar('movie_similar', book_id,
    #                             movie_similar_id, engine, metadata)
    #     print(f'기준 책: {book_id}, 유사 책: {rec_books}')


#윗부분처럼 함수 활용하여 바꿔보기
    for search_book_id in books_movies['id']:
    # 검색할 책 한 권만 포함한 table
        n_books_movies = create_new_movies(search_book_id, books_movies)

        document_embedding_list = get_document_vectors(
            n_books_movies['summary'], word2vec_model)
        
        cosine_similarities = cosine_similarity(
            document_embedding_list, document_embedding_list)
        
        rec_movies = recommendations(n_books_movies, search_book_id, cosine_similarities).id
        rec_movies = rec_movies.to_list()
        rec_movies = [x-max_id for x in rec_movies]

    print(f'현재 책: {search_book_id} 유사한 영화: {rec_movies}')
    print('추천 책(영화) 업데이트를 완료했습니다.')
    info_log('cb 유사도 업데이트 했습니다.')


except Exception as e:
    error_log(e)

