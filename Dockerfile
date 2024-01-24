FROM python:3.8

WORKDIR /usr/src/app

# Download data
RUN apt-get update && apt-get install -y curl
RUN mkdir ./data
RUN curl -LJO https://zenodo.org/records/10251194/files/cls_ent_idx.json && mv cls_ent_idx.json ./data/
RUN curl -LJO https://zenodo.org/records/10251194/files/ent_cls_idx.json && mv ent_cls_idx.json ./data/
# Uncomment if refs are needed in demo
#RUN curl -LJO https://zenodo.org/records/10251194/files/cls_ref_idx.json && mv cls_ref_idx.json ./data/
#RUN curl -LJO https://zenodo.org/records/10251194/files/ref_cls_idx.json && mv ref_cls_idx.json ./data/
COPY stopwords.txt ./stopwords.txt

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY ./src ./src
COPY favicon_zbMATH.png .
COPY Logo_zbMATH-Open-201223.svg .

# Run app
EXPOSE 8501
CMD [ "python", "./src/run_viewer.py" ]
#CMD ["streamlit", "run", "./src/viewer.py", "--server.port", "8501", "--server.address=172.17.0.2"]
