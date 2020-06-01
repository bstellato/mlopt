import mlopt.settings as stg


def error(err, error_type=ValueError):
    stg.logger.error(err)
    raise error_type(err)