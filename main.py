from cb import get_cosine_similarity, recommendations
from cf import get_similar_by_cf, predict_rating
from log import info_log, error_log
import pandas as pd
from datetime import datetime
from db import update_similarity, get_pd_from_table

now = datetime.now()
if now.hour == 3:
    # Contents Based 추천 시스템
    try:
        books = get_pd_from_table('books')
        books.drop(['isbn', 'pubDate', 'img', 'rate',
                   'bestseller'], axis=1, inplace=True)
        books.fillna('', inplace=True)
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

        cosine_similarities = get_cosine_similarity(books)
        info_log('cb 추천 모델을 업데이트 후 저장했습니다.')

        # similarity 값이 비어있는 경우만 업데이트
        non_similar_books = books[books['similarity'] == ''].index
        for index in non_similar_books:
            book_id = books.loc[index]['id']
            rec_books = recommendations(books, book_id, cosine_similarities).id
            rec_books = rec_books.to_list()
            rec_books = [str(i) for i in rec_books]
            rec_books = ",".join(rec_books)
            update_similarity('books', book_id, rec_books)
        info_log('cb 유사도 업데이트 했습니다.')
    except Exception as e:
        error_log(e)
else:
    # Collaborative Filtering 추천 시스템
    try:
        # 개인화 평점 계산해서 user-item 데이터셋으로 만들기
        item_sim_df, rating_matrix = get_similar_by_cf()
        ratings_pred = predict_rating(rating_matrix.values, item_sim_df.values)
        rating_pred_df = pd.DataFrame(
            ratings_pred, index=rating_matrix.index, columns=rating_matrix.columns)
        rating_pred_df = rating_pred_df.transpose()

        # 얻은 결과 저장 -> 추천에 사용
        rating_pred_df.to_csv('data/result_cf.csv', index=True)
        info_log('cf 추천이 업데이트 되었습니다.')
    except Exception as e:
        error_log(e)
