import pandas as pd
import numpy as np
from db import get_pd_from_table
from sklearn.metrics.pairwise import cosine_similarity
from log import info_log, error_log

try:
    ratings = pd.read_csv('data/watcha_ratings.csv', encoding='utf-8')
    ratings.dropna(axis=0, inplace=True)

    book_list = get_pd_from_table('book_list')
    book_list.drop(['review', 'status', 'created_dt',
                    'modified_dt'], axis=1, inplace=True)
    book_list.dropna(axis=0, inplace=True)

    # 실제 db에 있는 유저의 최대 id값 가져오기
    users = get_pd_from_table('users')
    max_user_id = users.id.max()

    # 유저 고유 id 부여
    ratings = ratings.sort_values(by='username', ascending=True)

    user_id = max_user_id + 1
    temp = None
    result_id = []
    for i in range(len(ratings)):
        if temp is None:
            temp = ratings.iloc[i]['username']
            user_id = user_id
            result_id.append(user_id)
        else:
            if temp == ratings.iloc[i]['username']:
                result_id.append(user_id)
            else:
                temp = ratings.iloc[i]['username']
                user_id += 1
                result_id.append(user_id)
    ratings['user_id'] = result_id
    # username isbn은 id 값으로 대체했으므로 삭제
    ratings.drop(['username', 'isbn'], axis=1, inplace=True)

    # ratings와 book_list 합치기
    ratings = pd.concat([ratings, book_list],
                        ignore_index=True, axis=0)
    ratings = ratings.sort_values(by='user_id', ascending=True)

    # pivot_table이용해서 user-item 데이터셋으로 만들기
    rating_matrix = ratings.pivot_table(
        values='rate', index='user_id', columns='book_id')
    # 유저가 매기지 않은 평점은 0으로 채우기
    rating_matrix = rating_matrix.fillna(0)

    # user-item 에서 item-user 데이터셋으로 변환하기 위해 Transpose()
    rating_matrix_T = rating_matrix.transpose()

    # 코사인 유사도 측정할 때, 매트릭스의 행 vector끼리 유사도 비교
    item_sim = cosine_similarity(rating_matrix_T, rating_matrix_T)
    item_sim_df = pd.DataFrame(
        item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)

    def predict_rating(ratings_arr, item_sim_arr):
        # 2차원 array로 만들어야 하므로 분모에 np.abs를 하나의 array안에 담기
        ratings_pred = ratings_arr.dot(
            item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis=1)])
        return ratings_pred

    # 개인화 평점 계산해서 user-item 데이터셋으로 만들기
    ratings_pred = predict_rating(
        rating_matrix.values, item_sim_df.values)
    rating_pred_df = pd.DataFrame(
        ratings_pred, index=rating_matrix.index, columns=rating_matrix.columns)
    rating_pred_df = rating_pred_df.transpose()

    # 얻은 결과 저장 -> 추천에 사용
    rating_pred_df.to_csv('data/result_cf.csv', index=True)
    info_log('cf 추천이 업데이트 되었습니다.')

except Exception as e:
    error_log(e)
