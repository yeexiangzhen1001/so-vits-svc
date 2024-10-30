import requests
import hashlib

# 读取文件内容并计算 MD5 值
with open("爸爸去哪儿加油板Vocal.wav", "rb") as file:
    file_bytes = file.read()
md5 = hashlib.md5(file_bytes).hexdigest()

# 构建请求参数和文件数据
params = {
    "group": "def",
    "md5": md5,
}
files = {
    "file": ("爸爸去哪儿加油板Vocal.wav", file_bytes),
}

# 构建签名
userId = "def"
signKey = "3610b3e6-3c43-424f-b29a-6ee4d2bf19b3"
sign = hashlib.md5(f"group={params['group']}&md5={md5}&key={signKey}".encode()).hexdigest().upper()
print(sign)
# 构建请求头
headers = {
    "sign": sign,
}

# 发送文件上传请求
response = requests.post("https://lecity.io/file/upload2", files=files, data=params, headers=headers)

# 处理响应
if response.status_code == 200:
    print("File uploaded successfully!")
    print(response.json())
else:
    print("Failed to upload file.")
    print(response.text)