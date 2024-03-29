FROM python:3.7

RUN apt-get update
RUN pip install mlflow
RUN pip install mysqlclient

RUN mkdir -p /home/mlflow
COPY src/run_server.sh /home/mlflow
WORKDIR /home/mlflow

ARG ARG_MLFLOW_ARTIFACT_URI
RUN mkdir -p /opt/mlflow_docker/mlflow_server/
RUN chmod 777 -R /opt/mlflow_docker/mlflow_server/

ENTRYPOINT ["sh", "/home/mlflow/run_server.sh"]