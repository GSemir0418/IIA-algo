# 使用官方Python 3.11.0镜像作为基础镜像
FROM python:3.11.0-slim

# 工作目录设置为/app
WORKDIR /app

# 复制当前目录下的内容到 /app 中
COPY . /app

# 安装pip依赖，从requirements.txt中读取，这里假设你已经创建了这个文件
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口，如果你的应用有Web接口
# EXPOSE 5432

# 运行你的应用，假设入口文件为app.py
# CMD ["python", "app.py"]