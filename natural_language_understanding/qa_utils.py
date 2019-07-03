import logging
import collections
import mxnet as mx
import gluonnlp as nlp
import bert
from mxnet.gluon.model_zoo import model_store

def download_qa_ckpt():
    model_store._model_sha1['bert_qa'] = '7eb11865ecac2a412457a7c8312d37a1456af7fc'
    result = model_store.get_model_file('bert_qa', root='./temp')
    print('Downloaded checkpoint to {}'.format(result))
    return result

def predict(dataset, all_results, vocab):
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=True)
    transform = bert.data.qa.SQuADTransform(tokenizer, is_pad=False, is_training=False, do_lookup=False)
    dev_dataset = dataset.transform(transform._transform)
    from bert.bert_qa_evaluate import PredResult, predict
    all_predictions = collections.OrderedDict()
    for features in dev_dataset:
        results = all_results[features[0].example_id]
    
        prediction, nbest = predict(
            features=features,
            results=results,
            tokenizer=nlp.data.BERTBasicTokenizer(lower=True))
    
        print('\nContext: %s\n'%(' '.join(features[0].doc_tokens)))
        question = features[0].input_ids.index('[SEP]')
        print('Question: %s\n'%(' '.join((features[0].input_ids[1:question]))))
        print('Top predictions: ')
        for i in range(3):
            print('%.2f%% \t %s'%(nbest[i][1] * 100, nbest[i][0]))
        print('')
