#FLAGS.max coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import code
import pdb
import numpy as np

from vizutils import *
import matplotlib.pyplot as pplot

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")
flags.DEFINE_bool("twotext", True, "In the case of the Dialogue act classification tasks, if sentence pairs are considered. ")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "diff_checkpoint", None,
    "Only when diffing checkpoints mode: A checkpoint to see the differences with the init_checkpoint (structure of graph from init_checkpoint)") 

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("nbatches" , 20, "number of batches to use for model inspection (make this lower if you are memory constrained) ")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")




tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines




class DailyDialogueProcessor(DataProcessor):
  """
  Processor for the Daily Dialogue data set.
  It implements at the moment only an utterance-pair task (feeding two utterances and classifying the second one).
  """

    
  def get_train_examples(self, data_dir):
    """See base class."""

    ff = self._ddread(data_dir,'train')
    return self._create_examples(ff, "train")

  def get_dev_examples(self, data_dir):
    """See base class."""

    ff = self._ddread(data_dir,'validation')
    return self._create_examples(ff, "validation")

  def get_test_examples(self, data_dir):
    """See base class."""
    ff = self._ddread(data_dir,'test')
    return self._create_examples(ff, "test")

  def get_labels(self):
    """See base class."""
    return ["Inform", "Question","Directive","Commisive"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue

      dialogue_a = [tokenization.convert_to_unicode(ll) for ll in line[1]][:-1]
      
      guid = "%s-%s" % (set_type, i)

      label = [tokenization.convert_to_unicode(ll) for ll in line[0]][:-1]

      #if set_type == "test":
      #  label = ["1" for mm in label]

      labels_list = self.get_labels()
      labels_map = {}
      for (i, lval) in enumerate(labels_list):
        labels_map[lval] = i

      if FLAGS.twotext:
        for k in range(1,len(dialogue_a)):
          examples.append(
            InputExample(guid=guid, text_a=dialogue_a[k-1], text_b=dialogue_a[k], label=labels_list[int(label[k])-1]))
      else:
        examples.append(
          InputExample(guid=guid, text_a=dialogue_a, text_b=None, label=label, dialogue = True))

    return examples

  @classmethod
  def _ddread(cls, data_dir, datadir_name):
    """
    reads the raw data for Daily Dialogue. 
    A seperate function creates the data as needed for the algorithm
    """
    with open(os.path.join(data_dir,datadir_name,"dialogues_act_"+datadir_name+'.txt'),'r') as f_acts:
      fa = f_acts.read()

    with open(os.path.join(data_dir, datadir_name,"dialogues_" + datadir_name + '.txt'),'r') as f_phrases:
      fp = f_phrases.read()

    rows_acts = fa.split('\n')
    rows_dialogues = fp.split('\n')
    
    return zip([r.split(' ') for r in rows_acts[:-1]][:-1], [r.split('__eou__') for r in rows_dialogues[:-1]][:-1])


class SWDAProcessor(DataProcessor):
    """
    This is customized by Dan Bondor to fit more easilly to the folder structure and file format that the GLUE benchmarks had.
    Processor for the Switchboard data set (https://web.stanford.edu/~jurafsky/ws97/CL-dialog.pdf).
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['sd', 'b', 'sv', '+', 'qy^d', 'na', 'x', '%', 'qo', 'ba', 'qy',
                'ny', 'h', 'aa', 'fc', 'ad', 'bh', 'nn', 'b^m', 'fo_o_fw_"_by_bc',
                'qh', 'bk', 'qw', 'bf', 'ng', '^2', 'no', 'arp_nd', 'qw^d',
                '^q', '^h', 'br', 'ar', 'qrr', 'ft', 'fp', 'bd', 't3', 't1',
                'oo_co_cc', '^g', 'fa', 'aap_am']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue # skip header row
            if i == 1:
                prev_line = line
                # If we need to classify using also the previous utterance, then 
                # we also skip the first example (since the beginning of the sentence is not modelled).
                if FLAGS.twotext:
                    continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(prev_line[3])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[4])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature, tokens


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings=True):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if FLAGS.do_train:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities, model)


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  tokens = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature, token = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
    tokens.append(token)
  return features




class InspectorBert():
  def __init__(self):

    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ddial": DailyDialogueProcessor,
        "swda": SWDAProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
      raise ValueError(
          "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
      raise ValueError(
          "Cannot use sequence length %d because the BERT model "
          "was only trained up to sequence length %d" %
          (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
      raise ValueError("Task not found: %s" % (task_name))

    self.processor = processors[task_name]()

    self.label_list = self.processor.get_labels()

    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    
    self.input_examples = self.processor.get_test_examples(FLAGS.data_dir)

    ## Create the actual model:

    ss = [FLAGS.train_batch_size, FLAGS.max_seq_length];

    self.input_ids_ = tf.placeholder(dtype = tf.int32, shape = ss, name = 'input_ids')
    self.input_mask_ = tf.placeholder(dtype = tf.int32, shape = ss, name = 'input_mask')
    self.segment_ids_ = tf.placeholder(dtype = tf.int32,shape= ss, name = 'segment_ids')
    self.label_id_ = tf.placeholder(dtype = tf.int32 , shape = [FLAGS.train_batch_size], name = 'label_id')

    (total_loss, per_example_loss, logits, probabilities, model) = create_model(
          bert_config, False, self.input_ids_, self.input_mask_, self.segment_ids_, self.label_id_,
          len(self.label_list), use_one_hot_embeddings = True)

    self.model = model
    self.total_loss = total_loss
    self.per_example_loss = per_example_loss
    self.logits = logits
    self.probabilities = probabilities


  def get_feed_dict_by_indices(self,example_indices):
    """
    input : a list or iterator of indices
    returns a feed_dict ready to be fed to the evaluator
    """
    input_feats_and_tokens = [convert_single_example(eid , self.input_examples[eid ],self.processor.get_labels(), FLAGS.max_seq_length, self.tokenizer) for eid  in example_indices]
    tokens = [inf[1] for inf in input_feats_and_tokens]
    input_feats = [inf[0] for inf in input_feats_and_tokens]
    input_ids  = np.array([iff.input_ids for iff in input_feats])
    input_mask = np.array([iff.input_mask for iff in input_feats])
    segment_ids = np.array([iff.segment_ids for iff in input_feats])
    label_id = np.array([iff.label_id for iff in input_feats])
    return {self.input_ids_ : input_ids, self.input_mask_:input_mask, self.segment_ids_ : segment_ids, self.label_id_ : label_id} 

  def get_tokens_by_indices(self, example_indices):
    input_feats_and_tokens = [convert_single_example(eid , self.input_examples[eid ],self.processor.get_labels(), FLAGS.max_seq_length, self.tokenizer) for eid  in example_indices]
    tokens = [inf[1] for inf in input_feats_and_tokens]
    return tokens



  def eval_batch(self,batch_num):
    """
    Evaluates certain parts of BERT for a batch (since all of the examples can't fit at once in the memory)
    """
    batch_size = FLAGS.train_batch_size
    examples = [r for r in range( batch_size*batch_num,batch_size+batch_size*batch_num)]
    feed_dict = self.get_feed_dict_by_indices(examples)
    

    # This is the only thing the classifier sees (bert is not really "pooling" - it is masking).
    a  = self.model.embedding_output.eval(feed_dict = feed_dict)

    pooled_out = self.model.get_pooled_output().eval(feed_dict = feed_dict)
    allprobs = [mm.eval(feed_dict = feed_dict) for mm in self.model.all_attention_probs]
    attention_heads  = [mm.eval(feed_dict = feed_dict) for mm in self.model.all_attention_heads_all_layers[-1]]# The raw attention heads from last layer before projecting to "hidden size". Hopefully they will show correlation with classes.
    attention_values = [mm.eval(feed_dict = feed_dict) for mm in self.model.all_attention_values_all_layers[-1]]# The raw attention heads from last layer before projecting to "hidden size". Hopefully they will show correlation with classes.


    tokens = self.get_tokens_by_indices(examples)

    return allprobs, pooled_out, attention_heads, attention_values, tokens

  def get_label_map(self):
    return {k[0] : k[1] for k in zip(self.label_list, range(0,len(self.label_list))) }

  def pca_attention_heads(self,allp, context_size = 768, ncomponents = 3):
    """
    returns the PCA scores and components of attention heads.
    The PCA *components* that correspond to PCA *scores* that separate well the classes,
    can be used to identify the (combination of) attention heads and attention layers that 
    contribute most to the separation of classes.

    allp : output from eval_batch (the 3rd element (index 2) of each list element corresponds to an attention head.)
    """

    from sklearn.decomposition import PCA
    att_heads = [a[2] for a in allp];
    att_outs = np.vstack([att[0].reshape([FLAGS.train_batch_size,-1,context_size]) for att in att_heads]);
    pp = att_outs.reshape([FLAGS.train_batch_size * FLAGS.nbatches ,-1]); 

    pca = PCA(n_components = ncomponents); # fewer can be better.
    pca.fit(pp)

    sc = pca.transform(pp); # The attention head vector projected in the PCA space
    #components = pca.components_;
    #ids_0 = np.where(np.array(class_ids) == 3)[0]; pplot.plot(sc[ids_0,0],sc[ids_0,1],'*');
    return sc, pca


  def get_example_class_int(self):
    """
    gets a list of integers corresponding to classes 
    for easier plotting. Assuming that the eval_batch was used
    """
    class_ids  = [self.get_label_map()[self.input_examples[k].label] for k in range(0,FLAGS.train_batch_size * FLAGS.nbatches)]
    return class_ids

  def gplotmatrix(self,sc, classinds):
    f = pplot.figure
    plt_ind = 1
    for k in range(0, sc.shape[1]):
      for m in range(0, sc.shape[1]):
        print("%i,%i"%(k,m))
        pplot.subplot(sc.shape[1],sc.shape[1],plt_ind)
        plt_ind = plt_ind + 1
        for _class in range(0,np.max(np.array(classinds))+1):
          cl_inds = np.where(np.array(classinds) == _class)[0]
          if k == m :
            pplot.hist(sc[cl_inds,k],20, alpha = 0.6, label = 'class %i'%_class)
            continue
          else:
            pplot.plot(sc[cl_inds,m],sc[cl_inds,k],"*", alpha = 0.6)
        if k == 0:
          pplot.title('Component %i'%m)
        if m == 0:
          pplot.ylabel('Component %i'%k)

        if plt_ind ==2 :
          pplot.legend(['class %i'%k for k in range(0,np.max(np.array(classinds))+1)])

    def scale_add_attention_probs(self,att_head_vector):
      """
      inner product attention to inner product attention:
      ---------------------------------------------------
      This should pin-point the attention heads (and attention head parts) 
      that contribute most to the attention head output being close to a vector that 
      looks like the input vector. This is performed 
      """




  @classmethod
  def check_changed_vars(self,ckpt1 = None,ckpt2 = None):
    """
    Takes two checkpoints, iterates over trainable variables
    and checks the differences in weights (made for BERT)
    """
    if ckpt1 == None:
      ckpt1 = FLAGS.init_checkpoint
     
    if ckpt2 == None:
      ckpt2 = FLAGS.diff_checkpoint

    with tf.Session() as sess:
      saver_all = tf.train.Saver()#sess,ckpt1)
      print("------------------")
      print("-  MODEL FROM:   -")
      print("------------------")
      print("-" + ckpt1  )
      print("------------------")
      saver_all.restore(sess,ckpt1)
      v = tf.trainable_variables(scope = 'bert')
      vars_to_check = [g for g in v if 'gamma' not in g.name and 'beta' not in g.name]
      v1 = [g.eval() for g in vars_to_check]
      saver2 = tf.train.Saver(vars_to_check)
      saver2.restore(sess,ckpt2)
      v2 = [g.eval() for g in vars_to_check]

      diffs = [np.sqrt(np.sum((mm[0]-mm[1])**2)) for mm in zip(v1, v2)]
        
      dd = [[v[0] for v in zip(diffs,vars_to_check) if 'layer_'+str(k) in v[1].name] for k in range(0,11)]; 
      #pplot.plot(dd[2:]);pplot.grid(True); pplot.title('Normalized Differences of BERT parameter layers after fine-tuning');pplot.legend(['layer '+str(k) for k in range(0,11)]);

    return diffs,dd,[v1,v2]










def main(_):
  bert_inspector  = InspectorBert()

  d1,d2,d3 = bert_inspector.check_changed_vars()

  with tf.Session() as sess:
    
    v1 =tf.report_uninitialized_variables().eval()
    saver = tf.train.Saver(); 
    saver.restore(sess,FLAGS.init_checkpoint)
    def restmod(n):
      saver.restore(sess,FLAGS.init_checkpoint[:-4]+str(n))


    # Evaluate the self attention matrices separately for each head and layer:
    allp = []
    for k in range(0,FLAGS.nbatches):
      allp.append(bert_inspector.eval_batch(k));

    

    tokens = [a[4] for a in allp]
    att_vals = [a[3] for a in allp]
    pooled_outs  = np.vstack([a[1] for a in allp])

    # get the attention heads separately for each batch simply by reshaping (batch @ first dimension should do that):
    att_heads = [a[2] for a in allp]
    att_outs = np.vstack([att[0].reshape([FLAGS.train_batch_size,-1,768]) for att in att_heads])

    att_probs = np.hstack([a[0] for a in allp]).swapaxes(0,1);

    # Compute a globally weighted attention matrix w.r.t. the second PCA component (best separation between first two classes):


    #p1 = model.all_attention_probs[11].eval(feed_dict = {input_ids_ : input_ids, input_mask_:input_mask, segment_ids_ : segment_ids, label_id_ : label_id});

    # plot PCA coefficients of the pooled output layer scores, to see if they separate well:
    sc,att_pca = bert_inspector.pca_attention_heads(allp)
    classes = bert_inspector.get_example_class_int()
    bert_inspector.gplotmatrix(sc,classes)
    #pplot.savefig('../results/PCA_components_attheads.png')

    v = att_pca.components_[1,:].reshape([12,60,-1])

    att_vals = np.vstack(att_vals).reshape([20*32, 12, 60, 64])

    B = att_pca.components_[1,:].reshape([-1,768])

    vars = dict(locals(),**globals());
    import rlcompleter
    import readline
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")

    head_probs = np.vstack([k[0] for k in allp]).reshape([32*20,12,12,60,60])



    code.interact(local = vars)



    from sklearn.decomposition import PCA
    pca = PCA(n_components = 10)
    pca.fit(pooled_outs)
    pp = pca.transform(pooled_outs)
    label_map = bert_inspector.get_label_map()
    class_ids  = [label_map[bert_inspector.input_examples[k].label] for k in range(0,pooled_outs.shape[0])]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot different classes with different symbols and colors in the scatterplot:
    cols = ['r','g','b','m']
    symbs = ['*','^','X','+']
    for _class in range(0,4):
      for kk in np.where(np.array(class_ids) == _class)[0]:
        ax.scatter(pp[kk,0],pp[kk,1],pp[kk,2],c  = cols[_class], label = bert_inspector.label_list[_class], edgecolors = 'none', marker = symbs[_class])


    ax.set_xlabel('c_1')
    ax.set_xlabel('c_2')
    ax.set_xlabel('c_3')

    pplot.title('Distribution of Principal Component scores \n Transformer Output')
    
    pplot.show()
    tt = 1; st = len(tokens[tt]) ; plot_attention_matrix(allp[0][tt,1,:,:],tokens[1], hide_special = True);pplot.show(block = False)








if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
