import logging
import os
import requests
import soundfile as sf
from flask import Flask, request, jsonify
from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map
from werkzeug.utils import secure_filename
import hashlib
import time

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# 配置文件路径
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def create_svc_model(model_path, config_path, cluster_model_path, enhance, diffusion_model_path, diffusion_config_path, shallow_diffusion, only_diffusion, use_spk_mix, feature_retrieval):
    return Svc(model_path,
               config_path,
               "cuda",  # 自动选择CPU或GPU
               cluster_model_path,
               enhance,
               diffusion_model_path,
               diffusion_config_path,
               shallow_diffusion,
               only_diffusion,
               use_spk_mix,
               feature_retrieval)

def upload_file_to_server(file_path, group="so-vits-svc", user_id="so-vits-svc", sign_key="3610b3e6-3c43-424f-b29a-6ee4d2bf19b3"):
    with open(file_path, "rb") as file:
        file_bytes = file.read()
    md5 = hashlib.md5(file_bytes).hexdigest()

    params = {
        "group": group,
        "md5": md5,
    }

    files = {
        "file": (os.path.basename(file_path), file_bytes),
    }

    sign = hashlib.md5(f"group={params['group']}&md5={md5}&key={sign_key}".encode()).hexdigest().upper()

    headers = {
        "sign": sign,
    }

    logging.info(f"Uploading file {file_path} to server...")
    response = requests.post("https://lecity.io/file/upload2", files=files, data=params, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        logging.info(f"Upload response: {response_json}")
        if response_json.get("code") == 200 and "data" in response_json:
            return response_json["data"].get("url")
    return None

@app.route('/convert', methods=['POST'])
def convert_audio():
    file = None
    file_path = None

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(f"File uploaded: {file_path}")
    elif 'url' in request.form and request.form['url'] != '':
        url = request.form['url']
        print(f"File download start from URL: {url}")
        start_time = time.time()
        response = requests.get(url)
        end_time = time.time()
        download_time = end_time - start_time
        print(f"File download completed in {download_time:.2f} seconds")

        if response.status_code == 200:
            filename = os.path.basename(url)
            file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"File downloaded from URL: {file_path}")
        else:
            print(f"Failed to download file from URL: {url}")
            return "Failed to download file", 400
    else:
        return "No file or URL provided", 400

    # 获取参数
    model_path = request.form.get('model_path', "logs/44k/main_model.pth")
    config_path = request.form.get('config_path', "configs/config.json")
    trans = request.form.get('trans', 0, type=int)
    spk = request.form.get('spk', 'speaker')
    slice_db = request.form.get('slice_db', -40, type=int)
    cluster_model_path = request.form.get('cluster_model_path', "logs/44k/kmeans_10000.pt")
    cluster_infer_ratio = request.form.get('cluster_infer_ratio', 0, type=float)
    auto_predict_f0 = request.form.get('auto_predict_f0', False, type=bool)
    noice_scale = request.form.get('noice_scale', 0.4, type=float)
    pad_seconds = request.form.get('pad_seconds', 0.5, type=float)
    clip = request.form.get('clip', 0, type=float)
    lg = request.form.get('linear_gradient', 0, type=float)
    lgr = request.form.get('linear_gradient_retain', 0.75, type=float)
    f0p = request.form.get('f0_predictor', 'pm')
    enhance = request.form.get('enhance', False, type=bool)
    enhancer_adaptive_key = request.form.get('enhancer_adaptive_key', 0, type=int)
    cr_threshold = request.form.get('f0_filter_threshold', 0.05, type=float)
    diffusion_model_path = request.form.get('diffusion_model_path', "logs/44k/diffusion/diffusion_model.pt")
    diffusion_config_path = request.form.get('diffusion_config_path', "logs/44k/diffusion/config.yaml")
    k_step = request.form.get('k_step', 100, type=int)
    shallow_diffusion = request.form.get('shallow_diffusion', False, type=bool)
    use_spk_mix = request.form.get('use_spk_mix', False, type=bool)
    second_encoding = request.form.get('second_encoding', False, type=bool)
    loudness_envelope_adjustment = request.form.get('loudness_envelope_adjustment', 1, type=float)
    only_diffusion = request.form.get('only_diffusion', False, type=bool)
    feature_retrieval = request.form.get('feature_retrieval', False, type=bool)
    # 新增 wav_format 参数
    wav_format = request.form.get('wav_format', 'wav')

    svc_model = create_svc_model(model_path, config_path, cluster_model_path, enhance, diffusion_model_path, diffusion_config_path, shallow_diffusion, only_diffusion, use_spk_mix, feature_retrieval)

    if len(spk_mix_map)<=1:
        use_spk_mix = False
    if use_spk_mix:
        spk = spk_mix_map

    raw_audio_path = file_path
    infer_tool.format_wav(raw_audio_path)

    kwarg = {
        "raw_audio_path" : raw_audio_path,
        "spk" : spk,
        "tran" : trans,
        "slice_db" : slice_db,
        "cluster_infer_ratio" : cluster_infer_ratio,
        "auto_predict_f0" : auto_predict_f0,
        "noice_scale" : noice_scale,
        "pad_seconds" : pad_seconds,
        "clip_seconds" : clip,
        "lg_num": lg,
        "lgr_num" : lgr,
        "f0_predictor" : f0p,
        "enhancer_adaptive_key" : enhancer_adaptive_key,
        "cr_threshold" : cr_threshold,
        "k_step": k_step,
        "use_spk_mix": use_spk_mix,
        "second_encoding": second_encoding,
        "loudness_envelope_adjustment": loudness_envelope_adjustment
    }

    audio = svc_model.slice_inference(**kwarg)
    key = "auto" if auto_predict_f0 else f"{trans}key"
    cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
    isdiffusion = "sovits"
    if shallow_diffusion:
        isdiffusion = "sovdiff"
    if only_diffusion:
        isdiffusion = "diff"
    if use_spk_mix:
        spk = "spk_mix"

    # 添加时间戳以避免文件名重复
    timestamp = int(time.time())
    res_filename = f'{filename}_{timestamp}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
    res_path = os.path.join(RESULT_FOLDER, res_filename)
    sf.write(res_path, audio, svc_model.target_sample, format=wav_format)

    logging.info(f"Generated audio file: {res_path}")

    svc_model.clear_empty()

    # 上传文件并获取下载链接
    download_url = upload_file_to_server(res_path)
    if download_url:
        logging.info(f"File uploaded successfully: {download_url}")
        return jsonify({"message": "File uploaded successfully!", "url": download_url, "code": 200})
    else:
        logging.error("Failed to upload file.")
        return jsonify({"message": "Failed to upload file.", "code": 500}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
