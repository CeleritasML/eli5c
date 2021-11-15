"""
Model Serving Demo

Usage:
    main.py [--host <host>] [--port <port>] [--debug]
    main.py (-h | --help)
    main.py --version

Options:
    --host <host>                     绑定的 Host [default: 0.0.0.0]
    --port <port>                     绑定的 Port [default: 9999]
    --debug                           是否开启 Debug [default: False]
    -h --help                         显示帮助
    -v --version                      显示版本

"""
import time

from docopt import docopt

import logging.handlers

date = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())

LOG_FILE = f'logs/eli5c_log_{date}.log'
log_format = '[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d, %(funcName)s] %(message)s'
logging.basicConfig(filename=LOG_FILE,
                    filemode='a',
                    format=log_format,
                    level=logging.INFO)
time_hdls = logging.handlers.TimedRotatingFileHandler(
  LOG_FILE, when='D', interval=1, backupCount=7)
logging.getLogger().addHandler(time_hdls)

logging.info('ELI5C Model begin service')

from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from models.eli5c_qa_model_resource import ELI5cQAModelResource


app = Flask(__name__)
CORS(app)

api = Api(app)
api.add_resource(ELI5cQAModelResource, '/v1/ELI5CModel')


if __name__ == '__main__':
    args = docopt(__doc__, version='Model Serving Demo v1.0.0')
    app.run(host=args['--host'], port=args['--port'], debug=args['--debug'])