import sqlalchemy as db
import pandas as pd
import os


def connect_db():
    """
    db 연결하기
    """
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
    engine = db.create_engine(DATABASE_URL)
    connection = engine.connect()
    metadata = db.MetaData()
    return engine, connection, metadata


def get_list_from_table(table_name):
    """
    table 이름을 입력 받아서 해당 데이터 전부 가져오기
    """
    engine, connection, metadata = connect_db()
    table = db.Table(table_name, metadata, autoload=True, autoload_with=engine)
    query = db.select([table])
    #  query = db.select([table]).where(table.columns.id == '1234')
    result = connection.execute(query)
    result_set = result.fetchall()
    return result_set


def get_pd_from_table(table_name):
    """
    table 이름을 입력 받아서 해당 데이터 dataframe으로 반환
    """
    engine, connection, metadata = connect_db()
    table = db.Table(table_name, metadata, autoload=True, autoload_with=engine)
    query = db.select([table])
    result_set = pd.read_sql_query(query, connection, index_col=None, coerce_float=True,
                                   params=None, parse_dates=None, chunksize=None, dtype=None)
    return result_set


if __name__ == '__main__':
    #  table_name = 'books'
    #  books = get_list_from_table(table_name)
    #  print(books[:5])

    table_name = 'book_list'
    book_list = get_list_from_table(table_name)
    print(book_list[:5])

    table_name = 'book_list'
    book_list = get_pd_from_table(table_name)
    print(book_list.head())
