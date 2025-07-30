# Base image
FROM registry.cn-shanghai.aliyuncs.com/memtensor/memos:amd-base

RUN rm -rf /app/

WORKDIR /app/

COPY . /app/

RUN #pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p /app/logs

# Set Python import path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Start the docker
CMD ["uvicorn", "memos.api.product_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
