# 基础镜像，使用官方 Python 3.10
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制打包配置和源码
COPY pyproject.toml README.md /app/
COPY pinhan/ /app/pinhan/

# 安装项目（pip install . 会自动安装依赖）
RUN pip install --upgrade pip \
    && pip install .

# 设置环境变量
ENV HOST=0.0.0.0
ENV PORT=3000

# 暴露端口
EXPOSE 3000

# 启动 API 服务
CMD ["pinhan-server"]