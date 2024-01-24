FROM python:3.9-slim

WORKDIR /app

# Download data
RUN apt-get update && apt-get install -y curl
RUN mkdir ./data
RUN curl -LJO https://zenodo.org/records/10251194/files/ent_cls_idx.json && mv ent_cls_idx.json ./data/

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app
COPY stopwords.txt ./
COPY favicon_zbMATH.png static/
COPY Logo_zbMATH-Open-201223.svg static/

# Command to run your application using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
