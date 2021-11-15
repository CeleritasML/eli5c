from flask_restful import Resource, request

from .eli5c_qa_model import ELI5cQAModel
from .validate_json import validate_json


model_instance = ELI5cQAModel()
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
        json = request.json
        return {'result': model_instance.ask(json['question'], 64)}
