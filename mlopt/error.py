import mlopt.settings as stg


def value_error(err, error_type=ValueError):
    stg.logger.error(err)
    raise error_type(err)


def warning(message):
    stg.logger.warning(message)
