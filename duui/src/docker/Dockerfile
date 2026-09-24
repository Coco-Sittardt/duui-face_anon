FROM python:3.8

WORKDIR /duui/python

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git\
    && rm -rf /var/lib/apt/lists/*


COPY python/requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY ./python .

#COPY ./python/custom_referencenet ./custom_referencenet
#COPY ./python/utils ./custom_referencenet
#COPY ./python/resources ./resources


EXPOSE 9714

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
