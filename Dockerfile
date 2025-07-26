FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install scikit-learn joblib

CMD ["python", "predict.py"]

