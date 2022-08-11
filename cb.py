import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
from tqdm import tqdm
import h5py
from db import get_pd_from_table
import zipfile


def get_cosine_similarity(books):
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
    word2vec_model = Word2Vec(size=200, window=5, min_count=2, workers=4)
    word2vec_model.build_vocab(corpus_list)
    word2vec_model.intersect_word2vec_format(
        'data/ko.bin.gz', lockf=1.0, binary=True)
    word2vec_model.train(
        corpus_list, total_examples=word2vec_model.corpus_count, epochs=20)

    document_embedding_list = get_document_vectors(
        books['summary'], word2vec_model)

    cosine_similarities = cosine_similarity(
        document_embedding_list, document_embedding_list)

    return cosine_similarities


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
        books = get_pd_from_table('books')
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

        cosine_similarities = get_cosine_similarity(books)
        print('내용 기반 유사도 모델 학습 완료')

        # save numpy array as h5 file
        h5f = h5py.File('data/result_cb.h5', 'w')
        h5f.create_dataset('similarity', data=cosine_similarities)
        h5f.close()
        print('유사도 모델 h5형식으로 저장 완료')

    except Exception as e:
        print(e)
