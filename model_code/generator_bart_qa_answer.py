import json
from time import time

import datasets
from tqdm import tqdm

from model_code.generator_bart_qa_train import load_support_doc, make_qa_s2s_model, make_qa_s2s_batch


def qa_s2s_generate_answers(
    question_and_doc,
    qa_s2s_model,
    qa_s2s_tokenizer,
    num_beams=8,
    min_len=16,
    max_len=256,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device='cuda:0',
):
    model_inputs = make_qa_s2s_batch([(question_and_doc, 'A')], qa_s2s_tokenizer, max_input_length, device=device,)
    generated_ids = []
    for i in [1, 2, 4]:
        generated_id = qa_s2s_model.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                min_length=min_len * i,
                max_length=max_len,
                do_sample=do_sample,
                early_stopping=True,
                num_beams=1 if do_sample else num_beams,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=qa_s2s_tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                decoder_start_token_id=qa_s2s_tokenizer.bos_token_id,
            )[0]
        generated_ids.append(generated_id)
    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]


def qa_s2s_generate(question_set, doc_set, qa_model, qa_tokenizer):
    qa_set = []
    for i in tqdm(range(len(question_set))):
        q = question_set[i]
        doc = doc_set[q['q_id']]
        question_doc = 'question: {} context: {}'.format(q['title'], doc)
        answers = qa_s2s_generate_answers(question_doc, qa_model, qa_tokenizer)
        q['bart_answers'] = answers
        qa_set.append(q)
    return qa_set


if __name__ == '__main__':
    st_time = time()
    eli5c = datasets.load_dataset('jsgao/eli5_category')
    eli5c_train_docs = load_support_doc('support_docs/eli5c_train_docs.dat')
    eli5c_val1_docs = load_support_doc('support_docs/eli5c_val1_docs.dat')
    eli5c_val2_docs = load_support_doc('support_docs/eli5c_val2_docs.dat')

    tokenizer, model = make_qa_s2s_model(from_file='models/eli5c_bart_model_2.pth')

    print('Start to generate answers', time() - st_time)

    answer_val1 = qa_s2s_generate(eli5c['validation1'], eli5c_val1_docs, model, tokenizer)
    print('Finish generating val 1 answers', time() - st_time)

    with open('answers/eli5-category-validation-1.json', 'w') as f:
        json.dump(answer_val1, f)
        print('Saved val 1 answers to %s' % f.name, time() - st_time)

    answer_val2 = qa_s2s_generate(eli5c['validation2'], eli5c_val2_docs, model, tokenizer)
    print('Finish generating val 2 answers', time() - st_time)

    with open('answers/eli5-category-validation-2.json', 'w') as f:
        json.dump(answer_val2, f)
        print('Saved val 2 answers to %s' % f.name, time() - st_time)

# 100%|██████████| 5446/5446 [4:52:06<00:00,  3.22s/it]
# Finish generating val 1 answers 17534.387187957764
# Saved val 1 answers to answers/eli5-category-validation-1.json 17534.68515944481
# 100%|██████████| 2375/2375 [2:03:50<00:00,  3.13s/it]
# Finish generating val 2 answers 24964.72337079048
# Saved val 2 answers to answers/eli5-category-validation-2.json 24964.849368333817
