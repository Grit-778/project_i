import os
from loguru import logger
import sys


# 确保logs目录存在
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, '..', 'logs')
logs_dir = os.path.abspath(logs_dir)
os.makedirs(logs_dir, exist_ok=True)


# 移除默认的日志处理器
logger.remove()


# 添加控制台输出（包含所有级别的日志）
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)


# 添加文件输出（包含所有级别的日志）
logger.add(
    os.path.join(logs_dir, "app.log"),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="500 MB",
    retention="10 days",
    encoding="utf-8"
)


# 移除了单独的error.log配置，所有日志都会记录在app.log中


# 导出配置好的logger实例
__all__ = ["logger"]