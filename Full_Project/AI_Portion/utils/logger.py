import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.Logger(name=name)
    log_format = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s (%(lineno)d): %(message)s"
    )

    file_handler = logging.FileHandler("BankAi.log", mode="a")
    file_handler.setFormatter(fmt=log_format)
    file_handler.setLevel(level=logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=log_format)
    stream_handler.setLevel(level=logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
