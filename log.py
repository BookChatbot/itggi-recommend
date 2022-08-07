import logging
import datetime
from pytz import timezone

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO, encoding='utf-8', filename='log/recommendation.log')


def info_log(message):
    logging.info(message)


def error_log(error_message):
    logging.error(error_message)


def get_log_date():
    dt = datetime.datetime.now(timezone("Asia/Seoul"))
    log_date = dt.strftime("%Y%m%d_%H:%M:%S")
    return log_date
