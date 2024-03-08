docker build -t IIA-algo-model-training-image .
docker run -it -p 5432:5432 --name IIA-algo-model-training-container IIA-algo-model-training-image