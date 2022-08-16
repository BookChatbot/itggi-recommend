from cf import get_similar_by_cf, predict_rating
from log import info_log, error_log
import pandas as pd
from db import delete_user_similar, get_pd_from_table, insert_user_similar, get_list_from_table, delete_book_similar, insert_book_similar, connect_db, get_not_update_pd_from_table
import h5py
import os

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
engine, connection, metadata = connect_db(DATABASE_URL)

try:
    users = get_pd_from_table('users', engine, connection, metadata)
    book_list = get_pd_from_table('book_list', engine, connection, metadata)
    print('데이터 불러오기 완료')

    # 개인화 평점 계산해서 user-item 데이터셋으로 만들기
    item_sim_df, rating_matrix = get_similar_by_cf(users, book_list)
    print('유사도 계산 완료')

    ratings_pred = predict_rating(rating_matrix.values, item_sim_df.values)
    rating_pred_df = pd.DataFrame(
        ratings_pred, index=rating_matrix.index, columns=rating_matrix.columns)
    rating_pred_df = rating_pred_df.transpose()
    print('점수 예측 테이블')
    print(rating_pred_df.head())

    # 유사도 유저별 추천 도서 저장
    user_ids = users['id']
    for user_id in user_ids:
        try:
            except_book_ids = get_list_from_table(
                'book_list', user_id, engine, connection, metadata)
            except_book_ids = [int(i[0]) for i in except_book_ids]
            print(f'유저가 저장한 책들: {except_book_ids} -> 제외 시키기')

            rating_pred_df.sort_values(
                user_id, ascending=False, inplace=True)
            # 유저가 저장한 책 보다 20개 더 많은 책을 가져오고 중복은 제외
            book_ids = list(rating_pred_df[:len(except_book_ids)+20].index)

            # 해당 유저의 원래 있던 추천 책들은 삭제
            delete_user_similar('user_similar', user_id, engine, metadata)

            # 중복되는 책은 제외하고 새로운 추천 책들 업데이트
            for book_id in book_ids:
                if book_id in except_book_ids:
                    print(f'{book_id} 는 겹치므로 제외')
                    continue
                print(f'유저기반 추천 결과: {user_id}, {book_id}')
                insert_user_similar(
                    'user_similar', user_id, book_id, engine, metadata)
        except:
            pass

    info_log('cf 추천이 업데이트 되었습니다.')
except Exception as e:
    error_log(e)


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


# Contents Based 추천 시스템
try:
    # 책 전체 불러오기
    print('books 테이블 불러오는 중...')
    books = get_pd_from_table('books', engine, connection, metadata)
    books.drop(['isbn', 'pubDate', 'img', 'rate',
                'bestseller'], axis=1, inplace=True)
    print(books.head())

    # 유사도가 없는 신간 책만 불러오기
    print('유사도 없는 책들만 불러오기...')
    non_books = get_not_update_pd_from_table(
        'books', 'book_similar', engine, connection, metadata)
    non_books = non_books['id']
    print(type(non_books), len(non_books))

    # load numpy array from h5 file
    h5f = h5py.File('data/result_cb.h5', 'r')
    cosine_similarities = h5f['similarity'][:]
    h5f.close()
    print('저장된 유사도 모델 h5 불러오기 완료')

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
    error_log(e)
