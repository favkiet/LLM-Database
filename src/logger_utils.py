import logging, os

os.makedirs("logs", exist_ok=True)

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = logging.FileHandler("logs/app.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger("llm_db")  # dùng tên cố định
logger.setLevel(logging.INFO)
logger.propagate = False              # quan trọng: tránh nhân bản do root

if not logger.handlers:               # chỉ gắn handler một lần
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)