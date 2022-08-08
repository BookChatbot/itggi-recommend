# start by pulling the python image
FROM python:3

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN mkdir log
RUN mkdir data
RUN pip install pip -U && pip install -r requirements.txt
RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python3" ]

CMD ["cf.py"]
