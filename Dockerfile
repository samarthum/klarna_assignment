FROM python:3.7

RUN pip install fastapi uvicorn sklearn xgboost pandas

COPY ./src /api/src

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]