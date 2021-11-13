import datasets
import pandas as pd

from model_code.generator_bart_qa_train import load_support_doc, make_qa_s2s_model, make_qa_s2s_batch

eli5c = datasets.load_dataset('jsgao/eli5_category')
eli5c_train_docs = load_support_doc('support_docs/eli5c_train_docs.dat')
eli5c_val1_docs = load_support_doc('support_docs/eli5c_val1_docs.dat')
eli5c_val2_docs = load_support_doc('support_docs/eli5c_val2_docs.dat')

tokenizer, model = make_qa_s2s_model(from_file='models/eli5c_bart_model_3.pth')
save_name = 'examples/bart_answer_epoch3.csv'


def qa_s2s_generate(
    question_and_doc,
    qa_s2s_model,
    qa_s2s_tokenizer,
    num_answers=1,
    num_beams=None,
    min_len=64,
    max_len=256,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device='cuda:0',
):
    model_inputs = make_qa_s2s_batch([(question_and_doc, 'A')], qa_s2s_tokenizer, max_input_length, device=device,)
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    generated_ids = qa_s2s_model.generate(
        input_ids=model_inputs['input_ids'],
        attention_mask=model_inputs['attention_mask'],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=qa_s2s_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=qa_s2s_tokenizer.bos_token_id,
    )
    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]


def gen_answers(dataset, indexes, docs, subset_name, model, tokenizer,
                results):
    for i in indexes:
        question = dataset[i]
        doc = docs[question['q_id']]
        # concatenate question and support document into BART input
        question_doc = 'question: {} context: {}'.format(question['title'], doc)
        # generate an answer with beam search
        answer1, answer2 = qa_s2s_generate(
            question_doc, model, tokenizer,
            num_answers=2,
            num_beams=8,
            min_len=64,
            max_len=256,
            max_input_length=512,
            device='cuda:0'
        )
        results['Question'] += [question['title']]
        results['Answer1'] += [answer1]
        results['Answer2'] += [answer2]
        results['Subset'] += [subset_name]


qa_results = {
    'Question': [],
    'Answer1': [],
    'Answer2': [],
    'Subset': [],
}

gen_answers(eli5c['train'], [12345, 15432, 51232, 57282], eli5c_train_docs, 'train', model, tokenizer, qa_results)
gen_answers(eli5c['validation1'], [0, 123, 3234], eli5c_val1_docs, 'val1', model, tokenizer, qa_results)
gen_answers(eli5c['validation2'], [2, 644, 1476], eli5c_val2_docs, 'val2', model, tokenizer, qa_results)


df = pd.DataFrame(qa_results)
df.to_csv(save_name, index=False)
