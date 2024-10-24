# 基于现有镜像
FROM registry.cn-beijing.aliyuncs.com/codewithgpu/svc-develop-team-so-vits-svc:hz91jlgLNj

# 复制本地文件到容器中的指定目录
COPY ./main.py /root/workdir/so-vits-svc/
COPY ./inference_main.py /root/workdir/so-vits-svc/

# 设置工作目录
WORKDIR /root/workdir/so-vits-svc/

# 指定容器启动时要运行的命令
CMD ["/root/miniconda3/bin/python", "main.py"]
