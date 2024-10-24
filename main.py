from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import List
import httpx
import uvicorn
import shutil
import asyncio
import sqlite3
import logging  # 导入 logging 模块

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # 指定日志文件名
    filemode='a'  # 以追加模式打开文件
)

import multiprocessing
import time
import os
import json
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor

import modules.commons as commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioCollate
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from modules.losses import (
    kl_loss,
    generator_loss, discriminator_loss, feature_loss
)

from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from multiprocessing import Value, Array

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# logging.getLogger('matplotlib').setLevel(logging.DEBUG)
# logging.getLogger('numba').setLevel(logging.DEBUG)

# 初始化 FastAPI 应用
app = FastAPI()

# # 全局变量，用于记录流程状态
# process_status = {
#     "current_task": "",
#     "progress": 0,
#     "message": "",
#     "error": None,
#     "total_epochs": 0,  # 总轮数
#     "current_epoch": 0  # 当前轮数
# }

# 在全局范围内定义共享内存变量
process_status = {
    "current_task": Array('c', b' ' * 256),  # 256字节的字符数组
    "progress": Value('i', 0),
    "message": Array('c', b' ' * 256),  # 256字节的字符数组
    "error": Array('c', b' ' * 256),  # 256字节的字符数组
    "total_epochs": Value('i', 0),
    "current_epoch": Value('i', 0)
}

# 创建一个锁
lock = mp.Lock()


# 初始化 SQLite 数据库
def init_db():
    conn = sqlite3.connect('process_status.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS process_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        current_task TEXT,
        progress INTEGER,
        message TEXT,
        error TEXT,
        total_epochs INTEGER,
        current_epoch INTEGER
    )
    ''')
    conn.commit()
    conn.close()


init_db()


# 插入或更新训练状态
def update_status(current_task=None, progress=None, message=None, error=None, total_epochs=None, current_epoch=None):
    conn = sqlite3.connect('process_status.db')
    cursor = conn.cursor()

    # 查询最新状态
    cursor.execute('SELECT * FROM process_status ORDER BY id DESC LIMIT 1')
    last_status = cursor.fetchone()

    # 如果数据库中没有状态记录，初始化一个默认值
    if not last_status:
        last_status = (None, "", 0, "", "", 0, 0)

    # 更新新状态，保留未提供参数的字段的旧值
    new_status = (
        current_task if current_task is not None else last_status[1],
        progress if progress is not None else last_status[2],
        message if message is not None else last_status[3],
        error if error is not None else last_status[4],
        total_epochs if total_epochs is not None else last_status[5],
        current_epoch if current_epoch is not None else last_status[6],
    )

    # 插入新状态
    cursor.execute('''
    INSERT INTO process_status (current_task, progress, message, error, total_epochs, current_epoch)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', new_status)
    conn.commit()
    conn.close()


# 获取最新状态
def get_latest_status():
    conn = sqlite3.connect('process_status.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM process_status ORDER BY id DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    return row


# 更新数据库中的当前轮数
def update_current_epoch(epoch):
    conn = sqlite3.connect('process_status.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE process_status SET current_epoch = ? WHERE id = (SELECT MAX(id) FROM process_status)',
                   (epoch,))
    conn.commit()
    conn.close()


train_download_dir = "/root/workdir/audio-slicer/input"
val_download_dir = "/root/workdir/so-vits-svc/raw"

executor = ThreadPoolExecutor(max_workers=10)


# 定义训练参数的模型
class TrainingParameters(BaseModel):
    vocoder_name: str = Field(default="nsf-hifigan", description="选择一种声码器")
    log_interval: int = Field(default=200, description="多少步输出一次日志")
    eval_interval: int = Field(default=800, description="多少步进行一次验证并保存一次模型")
    epochs: int = Field(default=10000, description="训练总轮数，达到此轮数后将自动停止训练")
    learning_rate: float = Field(default=0.001, description="学习率，建议保持默认值不要改")
    batch_size: int = Field(default=32, description="单次训练加载到 GPU 的数据量")
    all_in_mem: bool = Field(default=False, description="是否将所有数据集加载到内存中")
    keep_ckpts: int = Field(default=3, description="训练时保留最后几个模型")


# 定义请求体模型
class TrainRequest(BaseModel):
    parameters: TrainingParameters
    train_dataset: List[str] = Field(..., description="训练集文件列表")
    val_dataset: List[str] = Field(..., description="验证集文件列表")


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, index):
        return self.__dict__.get(index)


def get_hparams(config_path="./configs/config.json", model_name='44k', init=True):
    if model_name is None:
        raise ValueError("Model name must be provided.")

    model_dir = os.path.join("./logs", model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()

    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def run(rank, n_gpus, hps):
    global global_step

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # for pytorch on win, backend use gloo
    dist.init_process_group(backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus,
                            rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem  # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem, vol_aug=False)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank])

    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step = int(name[name.rfind("_") + 1:name.rfind(".")]) + 1
        # global_step = (epoch_str - 1) * len(train_loader)
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # 更新总轮数
        logging.info(epoch / hps.train.epochs)
        update_status("Training started...", epoch / hps.train.epochs * 100, "Training in progress", "", hps.train.epochs,
                      epoch)

        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None)
        # update learning rate
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv, volume = items
        g = spk.cuda(rank, non_blocking=True)
        spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, ids_slice, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g,
                                                                                    c_lengths=lengths,
                                                                                    spec_lengths=lengths, vol=volume)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                reference_loss = 0
                for i in losses:
                    reference_loss += i
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(
                    f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                        pred_lf0[0, 0, :].detach().cpu().numpy()),
                    "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                             norm_lf0[0, 0, :].detach().cpu().numpy())
                }

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        logger.info(f'当前轮数: {process_status["current_epoch"].value}/{process_status["total_epochs"].value}')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv, volume = items
            g = spk[:1].cuda(0)
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv = uv[:1].cuda(0)
            if volume != None:
                volume = volume[:1].cuda(0)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_hat, _ = generator.module.infer(c, f0, uv, g=g, vol=volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            f"gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


# 下载文件的异步函数
async def download_file(url: str, directory: str):
    filename = os.path.join(directory, url.split("/")[-1])
    async with httpx.AsyncClient() as client:
        update_status(f"Downloading {filename}...", 0, f"Downloading {filename}...", "", 0, 0)
        logging.info(f"Downloading {filename}...")
        try:
            response = await client.get(url)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
            update_status(f"Downloaded {filename}", 0, f"Downloaded {filename}", "", 0, 0)
            logging.info(f"Downloaded {filename}")
        except Exception as e:
            update_status(f"Error downloading {filename}: {str(e)}", 0, "Download failed", str(e), 0, 0)
            logging.error(f"Error downloading {filename}: {str(e)}")


# 配置文件按需求修改
def update_config_file(parameters: TrainingParameters, config_path: str):
    update_status("Updating configuration file...", 0, f"Updating config at {config_path}...", "", 0, 0)
    logging.info(f"Updating config at {config_path}...")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        config['train']['log_interval'] = parameters.log_interval
        config['train']['eval_interval'] = parameters.eval_interval
        config['train']['epochs'] = parameters.epochs
        config['train']['learning_rate'] = parameters.learning_rate
        config['train']['batch_size'] = parameters.batch_size
        config['train']['all_in_mem'] = parameters.all_in_mem
        config['train']['keep_ckpts'] = parameters.keep_ckpts
        config['model']['vocoder_name'] = parameters.vocoder_name

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        update_status("Configuration file updated successfully.", 0, "Configuration file updated successfully.", "", 0,
                      0)
        logging.info("Configuration file updated successfully.")
    except Exception as e:
        update_status(f"Error updating config file: {str(e)}", 0, "Configuration update failed", str(e), 0, 0)
        logging.error(f"Error updating config file: {str(e)}")


async def read_output(stream, log_func):
    while True:
        line = await stream.readline()
        if not line:
            break
        log_func(line.decode().strip())


async def train_model(config_path: str, parameters: TrainingParameters):
    """训练主模型"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    # 获取超参数
    hps = get_hparams(config_path="./configs/config.json", model_name="44k")

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    update_status("开始主模型训练...", 0, "开始主模型训练...", "", 0, 0)
    logging.info("开始主模型训练...")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, lambda: mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps)))


# 训练主模型
# async def train_model(config_path: str):
#     command = f"python train.py -c {config_path} -m 44k"
#     with lock:
#         process_status["current_task"] = "开始主模型训练..."
#         logging.info(process_status["current_task"])
#
#     process = await asyncio.create_subprocess_shell(
#         command,
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE,
#         cwd="/root/workdir/so-vits-svc"
#     )
#
#     # 启动输出读取任务
#     await asyncio.gather(
#         read_output(process.stdout, lambda output: logging.info(output)),
#         read_output(process.stderr, lambda output: logging.error(output))
#     )
#
#     return_code = await process.wait()
#     if return_code == 0:
#         with lock:
#             process_status["message"] = "模型训练成功完成！"
#             process_status["progress"] = 100
#             process_status["error"] = None
#             logging.info(process_status["message"])
#     else:
#         with lock:
#             process_status["error"] = f"模型训练失败，退出码：{return_code}"
#             process_status["message"] = "模型训练失败"
#         logging.error(process_status["error"])


async def inference_model(model_path: str, config_path: str, audio_name: str, speaker: str):
    """根据训练出来的模型进行推理"""
    command = f"python inference_main.py -m \"{model_path}\" -c \"{config_path}\" -n \"{audio_name}\" -t 0 -s \"{speaker}\""
    logging.info(command)

    update_status("Running inference...", None, "Running inference...", "", None, None)
    logging.info("Running inference...")

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        logging.info(f"Inference output: {stdout.decode()}")

        update_status(f"Inference completed for model: {model_path} and audio: {audio_name}", None,
                      f"Inference completed for model: {model_path} and audio: {audio_name}", "", None, None)

    except Exception as e:
        update_status(f"Error during inference for model: {model_path} and audio: {audio_name}: {str(e)}", None,
                      "Inference failed", str(e), None, None)
        logging.error(f"Error during inference for model: {model_path} and audio: {audio_name}: {str(e)}")


# 预处理
async def pre_processing(output_dir: str, input_dir: str, work_dir: str, parameters: TrainingParameters):
    command = f"python ./audio-slicer.py --output {output_dir} --input {input_dir} 10"
    update_status("Running audio slicer...", 0, "Running audio slicer...", "", 0, 0)
    logging.info("Running audio slicer...")

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir
        )

        stdout, stderr = await process.communicate()
        logging.info(f"Audio slicer output: {stdout.decode()}")
        update_status("Audio slicer completed!", 0, "Audio slicer completed!", "", 0, 0)

        # 创建 config.json
        config_content = (
            '"n_speakers": 1\n\n'
            '"spk":{\n'
            '    "speaker": 0,\n'
            '}'
        )
        config_path = "/root/workdir/so-vits-svc/dataset_raw/config.json"

        # 创建 config.json 文件并写入内容
        with open(config_path, "w") as config_file:
            config_file.write(config_content)

        logging.info(f"Created config.json at {config_path}")

        # 创建 speaker 目录
        speaker_dir = "/root/workdir/so-vits-svc/dataset_raw/speaker"
        os.makedirs(speaker_dir, exist_ok=True)

        # 移动文件到 speaker 目录
        output_dir = "/root/workdir/audio-slicer/output"
        output_files = os.listdir(output_dir)
        for file_name in output_files:
            src_file = os.path.join(output_dir, file_name)
            dst_file = os.path.join(speaker_dir, file_name)
            shutil.move(src_file, dst_file)

        update_status("Files moved to speaker directory and config.json created!", 0,
                      "Files moved to speaker directory and config.json created!", "", 0, 0)
        logging.info("Files moved to speaker directory and config.json created!")

        # 执行重采样
        resample_command = "python resample.py --skip_loudnorm"
        update_status("Resampling audio files to 44100Hz mono...", 0, "Resampling audio files to 44100Hz mono...", "",
                      0, 0)
        logging.info("Resampling audio files to 44100Hz mono...")

        try:
            resample_process = await asyncio.create_subprocess_shell(
                resample_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/root/workdir/so-vits-svc"
            )

            stdout, stderr = await resample_process.communicate()
            logging.info(f"Resampling output: {stdout.decode()}")
            update_status("Resampling completed!", 0, "Resampling completed!", "", 0, 0)
            logging.info("Resampling completed!")
        except Exception as e:
            update_status(f"Error running resampling: {str(e)}", 0, "Resampling failed", str(e), 0, 0)
            logging.error(f"Error running resampling: {str(e)}")

        # 自动划分训练集和验证集，并生成配置文件
        preprocess_command = "python preprocess_flist_config.py --speech_encoder vec768l12"
        update_status("Splitting dataset and generating configuration...", 0,
                      "Splitting dataset and generating configuration...", "", 0, 0)
        logging.info("Splitting dataset and generating configuration...")

        try:
            preprocess_process = await asyncio.create_subprocess_shell(
                preprocess_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/root/workdir/so-vits-svc"
            )

            stdout, stderr = await preprocess_process.communicate()
            logging.info(f"Preprocessing output: {stderr.decode()}")
            update_status("Dataset splitting and configuration generation completed!", 0,
                          "Dataset splitting and configuration generation completed!", "", 0, 0)
            logging.info("Dataset splitting and configuration generation completed!")
        except Exception as e:
            update_status(f"Error running preprocessing: {str(e)}", 0, "Preprocessing failed", str(e), 0, 0)
            logging.error(f"Error running preprocessing: {str(e)}")

        # 更新配置文件
        config_path = "/root/workdir/so-vits-svc/configs/config.json"
        update_config_file(parameters, config_path)

        # 生成 Hubert 与 F0
        hubert_f0_command = "python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 8"
        update_status("Generating Hubert and F0...", 0, "Generating Hubert and F0...", "", 0, 0)
        logging.info("Generating Hubert and F0...")

        try:
            hubert_f0_process = await asyncio.create_subprocess_shell(
                hubert_f0_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/root/workdir/so-vits-svc"
            )

            stdout, stderr = await hubert_f0_process.communicate()
            logging.info(f"Hubert and F0 generation output: {stderr.decode()}")
            update_status("Hubert and F0 generation completed!", 0, "Hubert and F0 generation completed!", "", 0, 0)
            logging.info("Hubert and F0 generation completed!")
        except Exception as e:
            update_status(f"Error generating Hubert and F0: {str(e)}", 0, "Hubert and F0 generation failed", str(e), 0,
                          0)
            logging.error(f"Error generating Hubert and F0: {str(e)}")

        # 在预处理完成后调用训练函数
        config_path = "/root/workdir/so-vits-svc/configs/config.json"
        await train_model(config_path, parameters)

        # 调用推理函数，传入模型路径、配置路径、音频文件名和说话者
        model_directory = "logs/44k/"
        model_files = get_pth_files(model_directory)

        audio_directory = "raw/"
        audio_files = os.listdir(audio_directory)

        speaker = "speaker"

        for model_file in model_files:
            model_path = os.path.join(model_directory, model_file)

            for audio_path in audio_files:
                await inference_model(model_path, config_path, audio_path, speaker)

                try:
                    os.remove(f'raw/{audio_path}')
                    logging.info(f"Deleted audio file: raw/{audio_path}")
                except Exception as e:
                    logging.error(f"Error deleting audio file raw/{audio_path}: {str(e)}")

        logging.info("SUCCESS!!!")
    except Exception as e:
        update_status(f"Error running audio slicer: {str(e)}", 0, "Audio slicing failed", str(e), 0, 0)
        logging.error(f"Error running audio slicer: {str(e)}")


# 数据预处理
async def data_pre_processing(train_files: List[str], val_files: List[str], train_download_dir: str,
                              val_download_dir: str, parameters: TrainingParameters):
    total_files = len(train_files) + len(val_files)

    update_status("Downloading training files...", 0, "Started downloading training files", "", 0, 0)
    logging.info("Started downloading training files")

    for i, file_url in enumerate(train_files):
        await download_file(file_url, train_download_dir)

    update_status("Downloading validation files...", 0, "Started downloading validation files", "", 0, 0)
    logging.info("Started downloading validation files")

    for i, file_url in enumerate(val_files):
        await download_file(file_url, val_download_dir)

    update_status("Download complete! Starting audio slicing...", 0, "All files downloaded, proceeding to slicing", "",
                  0, 0)
    logging.info("All files downloaded, proceeding to slicing")
    await pre_processing(output_dir="output", input_dir="input", work_dir="/root/workdir/audio-slicer",
                         parameters=parameters)


def get_pth_files(directory: str):
    pth_files = []
    for file in os.listdir(directory):
        if file.endswith('.pth') and file.startswith('G_'):
            pth_files.append(file)
    return pth_files


def get_all_files(directory: str):
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]


def get_pth_files(directory: str):
    """获取指定目录下以 G_ 开头的所有 .pth 文件"""
    pth_files = []
    for file in os.listdir(directory):
        if file.endswith('.pth') and file.startswith('G_'):
            pth_files.append(file)
    return pth_files


def get_all_files(directory: str):
    """获取指定目录下的所有文件"""
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]


# 获取阶段列表
@app.get("/model_checkpoints")
async def list_pth_files():
    """API接口返回.pth文件列表"""
    directory = "logs/44k/"
    pth_files = get_pth_files(directory)
    return {"code": 200, "files": pth_files}


# 下载阶段模型
@app.get("/download_checkpoint/{file_name}")
async def download_pth(file_name: str):
    directory = "logs/44k/"
    file_path = os.path.join(directory, file_name)
    return FileResponse(file_path)


# 下载日志
@app.get("/download_log")
async def download_log():
    log_file_path = "app.log"
    if os.path.exists(log_file_path):
        return FileResponse(log_file_path)
    else:
        return {"error": "Log file not found."}


@app.get("/logs")
async def get_logs(limit: int = Query(10, ge=1)):
    log_file_path = "app.log"

    if not os.path.exists(log_file_path):
        return {"error": "Log file not found."}

    try:
        with open(log_file_path, "r") as log_file:
            # 读取所有行
            lines = log_file.readlines()
            # 返回最后 limit 行
            return PlainTextResponse(''.join(lines[-limit:]))
    except Exception as e:
        return {"error": str(e)}


# 处理训练请求
@app.post("/set_training_params")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    if request.parameters.vocoder_name not in ["nsf-hifigan", "nsf-snake-hifigan"]:
        return {"code": 400, "message": f"Invalid vocoder_name: {request.parameters.vocoder_name}"}

    # 启动后台任务
    background_tasks.add_task(data_pre_processing, request.train_dataset, request.val_dataset, train_download_dir,
                              val_download_dir, request.parameters)
    logging.info("Training started")  # 添加调试日志
    return {"code": 200, "message": "Training started, files are being downloaded and processed."}


# 获取流程状态
@app.get("/status")
async def get_status():
    # 获取最新状态
    status = get_latest_status()

    # 设置默认值
    default_status = {
        "current_task": "No task running",
        "progress": 0,
        "message": "No message",
        "error": "",
        "total_epochs": 0,
        "current_epoch": 0
    }

    if status:
        # 解包状态记录，如果存在的话
        current_task, progress, message, error, total_epochs, current_epoch = status[1:]
        return {
            "current_task": current_task,
            "progress": progress,
            "message": message,
            "error": error,
            "total_epochs": total_epochs,
            "current_epoch": current_epoch
        }
    else:
        # 返回默认状态
        return default_status


# 开始训练
@app.post("/start_training")
async def start_training(parameters: TrainingParameters, background_tasks: BackgroundTasks):
    # 将训练任务添加到后台任务中
    background_tasks.add_task(train_model, "configs/config.json", parameters)
    return {"code": 200, "message": "训练已开始"}


# 获取结果文件列表
@app.get("/results")
async def list_result_files():
    """API接口返回结果文件列表"""
    directory = "results/"  # 指定结果文件的目录
    result_files = get_all_files(directory)
    return {"code": 200, "files": result_files}


# 下载结果文件
@app.get("/download_result/{file_name}")
async def download_result(file_name: str):
    directory = "results/"  # 指定结果文件的目录
    file_path = os.path.join(directory, file_name)
    return FileResponse(file_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
