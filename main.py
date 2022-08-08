from cb import get_cosine_similarity, get_document_vectors, recommendations
from cf import get_similar_by_cf, predict_rating
from log import info_log, error_log
import pandas as pd
from datetime import datetime
from db import update_similarity

now = datetime.now()
if now.hour == 23:
    # Contents Based 추천 시스템
    try:
        cosine_similarities, books, word2vec_model = get_cosine_similarity()
        document_embedding_list = get_document_vectors(
            books['summary'], word2vec_model)
        books.drop(['summary'], axis=1, inplace=True)

        for book_id in books['id']:
            rec_books = recommendations(books, book_id, cosine_similarities).id
            rec_books = rec_books.to_list()
            rec_books = [str(i) for i in rec_books]
            rec_books = ",".join(rec_books)
            update_similarity('books', book_id, rec_books)
        info_log('cb 추천이 업데이트 되었습니다.')
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
