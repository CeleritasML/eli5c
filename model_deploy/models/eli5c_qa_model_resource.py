import logging

from flask_restful import Resource, request

from .eli5c_qa_model import ELI5cQAModel
from .validate_json import validate_json


model_instance = ELI5cQAModel()

logging.info('ELI5C Model Resource Loaded.')

model_schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'}
    },
    'required': ['question']
}


class ELI5cQAModelResource(Resource):
    @validate_json(model_schema)
    def post(self):
        question = request.json['question']
        logging.info('[Question received]: %s' % question)
        answer = model_instance.ask(question, 64)
        logging.info('[Answer generated]: %s' % answer)
        return {'result': answer}
