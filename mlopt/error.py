import mlopt.settings as stg


def value_error(err):
    stg.logger.error(err)
    raise ValueError(err)
