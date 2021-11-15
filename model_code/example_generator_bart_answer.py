import datasets
import pandas as pd

from model_code.generator_bart_qa_answer import qa_s2s_generate_answers
from model_code.generator_bart_qa_train import load_support_doc, make_qa_s2s_model

eli5c = datasets.load_dataset('jsgao/eli5_category')
eli5c_train_docs = load_support_doc('support_docs/eli5c_train_docs.dat')
eli5c_val1_docs = load_support_doc('support_docs/eli5c_val1_docs.dat')
eli5c_val2_docs = load_support_doc('support_docs/eli5c_val2_docs.dat')

tokenizer, model = make_qa_s2s_model(from_file='models/eli5c_bart_model_0.pth')
save_name = 'examples/bart_answer_epoch0.csv'


def gen_answers(dataset, indexes, docs, subset_name, qa_model, qa_tokenizer, results):
    for i in indexes:
        question = dataset[i]
        doc = docs[question['q_id']]
        # concatenate question and support document into BART input
        question_doc = 'question: {} context: {}'.format(question['title'], doc)
        # generate an answer with beam search
        answer1, answer2, answer3 = qa_s2s_generate_answers(question_doc, qa_model, qa_tokenizer)
        results['Question'] += [question['title']]
        results['Answer1'] += [answer1]
        results['Answer2'] += [answer2]
        results['Answer3'] += [answer3]
        results['Subset'] += [subset_name]


qa_results = {
    'Question': [],
    'Answer1': [],
    'Answer2': [],
    'Answer3': [],
    'Subset': [],
}

gen_answers(eli5c['train'], [12345, 15432, 51232, 57282], eli5c_train_docs, 'train', model, tokenizer, qa_results)
gen_answers(eli5c['validation1'], [0, 123, 3234], eli5c_val1_docs, 'val1', model, tokenizer, qa_results)
gen_answers(eli5c['validation2'], [2, 644, 1476], eli5c_val2_docs, 'val2', model, tokenizer, qa_results)


df = pd.DataFrame(qa_results)
df.to_csv(save_name, index=False)
