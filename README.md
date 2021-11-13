# ABSA_Project

* transformers exercise: https://colab.research.google.com/drive/1kZ42tj34UiIWABgP1tdZzDdeQ05vz5Kf#scrollTo=KkowGjBrWMw6
* current issue: https://www.google.com/search?q=TypeError%3A+dropout()%3A+argument+%27input%27+(position+1)+must+be+Tensor%2C+not+list&oq=TypeError%3A+dropout()%3A+argument+%27input%27+(position+1)+must+be+Tensor%2C+not+list&aqs=edge..69i57j69i58.901j0j1&sourceid=chrome&ie=UTF-8

* Run cmd: 
  * rmdir /s results \
               (after each run)
  * python run_classifier_TABSA.py --task_name sentihood_NLI_M --data_dir data/sentihood/bert-pair/ --vocab_file uncased_L-12_H-768_A-12/vocab.txt --bert_config_file uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin --eval_test --do_lower_case --max_seq_length 512 --train_batch_size 6 --learning_rate 2e-5 --num_train_epochs 6.0 --output_dir results/sentihood/NLI_M --seed 42\
          (may need to adjust --num_train_epochs 6.0)
