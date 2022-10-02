from cf import get_similar_by_cf, predict_rating
from log import info_log, error_log
import pandas as pd
from db import delete_user_similar, get_pd_from_table, insert_user_similar, get_list_from_table, connect_db
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
