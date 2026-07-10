FROM python:3.8

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY src/python/requirements.txt src/python/requirements.txt
RUN pip install -r ./src/python/requirements.txt




EXPOSE 8001

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
CMD ["--workers", "1"]