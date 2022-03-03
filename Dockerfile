FROM python:3.7

RUN pip install fastapi uvicorn scikit-learn xgboost pandas

COPY ./src /api/src

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["src.app:app", "--host", "0.0.0.0"]