FROM python:3.11.4-bookworm

RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.25.2
RUN pip3 install pandas==2.0.3
RUN pip3 install ipykernel
RUN pip3 install seaborn==0.12.2
RUN pip3 install scikit-learn==1.3.0
RUN pip3 install ppscore
RUN pip3 install shap
RUN pip3 install dash
RUN pip3 install dash_bootstrap_components
RUN pip3 install mlflow

WORKDIR /root/source_code/app

COPY ./source_code/app /root/source_code/app

# CMD tail -f /dev/null

