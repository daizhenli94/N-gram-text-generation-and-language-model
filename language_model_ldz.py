from collections import *
from random import random
import pprint
import operator
import math
import mixem
import numpy as np

def print_probs(lm, history):
    probs = sorted(lm[history],key=lambda x:(-x[1],x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)

def train_char_lm(fname, order=4, add_k=1):
  ''' Trains a language model.

  This code was borrowed from 
  http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

  Inputs:
    fname: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED

  Returns:
    A dictionary mapping from n-grams of length n to a list of tuples.
    Each tuple consists of a possible net character and its probability.
  '''

  # TODO: Add your implementation of add-k smoothing.

  data = open(fname,'r',encoding='utf8',errors='replace').read()
  lm = defaultdict(Counter)
  pad = "~" * order
  data = pad + data
  #add_k smoothing ----------------
  vocab = []
  for i in range(len(data)-order):
    vocab.append(data[i:i+order])
  vocab = set(vocab)
  vocab = list(vocab)
  v = len(vocab) #number of vocabularies
  vocabs = set(list(data))
  vocabs = list(vocabs)
  number = len(vocabs)
  #--------------------------------------
  for i in range(len(data)-order):
    history, char = data[i:i+order], data[i+order]
    lm[history][char]+=1
  #for i in range(len(vocab)):
      #history, char = vocab[i], data[i+orders]
      #lm[history][char] = lm[history][char]+add_k
  def normalize(counter):
    #s = float(sum(counter.values()))
    s = float(sum(counter.values())+add_k*v)
    #return [(c,cnt/s) for c,cnt in counter.items()]
    return [(c,(cnt+add_k)/s) for c,cnt in counter.items()]
  outlm = {hist:normalize(chars) for hist, chars in lm.items()}
  return outlm


def generate_letter(lm, history, order):
  ''' Randomly chooses the next letter using the language model.
  
  Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of text at least 'order' long.
    order: The length of the n-grams in the language model.
    
  Returns: 
    A letter
  '''
  
  history = history[-order:]
  dist = lm[history]
  x = random()
  for c,v in dist:
    x = x - v
    if x <= 0: return c
    
    
def generate_text(lm, order, nletters=500):
  '''Generates a bunch of random text based on the language model.
  
  Inputs:
  lm: The output from calling train_char_lm.
  history: A sequence of previous text.
  order: The length of the n-grams in the language model.
  
  Returns: 
    A letter  
  '''
  history = "~" * order
  out = []
  for i in range(nletters):
    c = generate_letter(lm, history, order)
    history = history[-order:] + c
    out.append(c)
  return "".join(out)

def perplexity(test_filename, lm, order=1):
  '''Computes the perplexity of a text file given the language model.
  
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model.
  '''
  test = open(test_filename,'r',encoding='utf8',errors='replace').read()
  pad = "~" * order
  test = pad + test
  # TODO: YOUR CODE HRER
  N = len(test)-order
  sum_probability = 0
  for i in range(len(test)-order):
    wi = test[i+order]
    wn = test[i:i+order]
    #-----uncomment these if not using back up-------
    #if wn in lm:
      #for j in range(len(lm[wn])):
        #if lm[wn][j][0] == wi:
          #probability = math.log((1/lm[wn][j][1]))
          #sum_probability = sum_probability+probability
          #N += 1
    #--------------------------------------------------
    #------comment these if not using back up-----------
    lambdas = set_lambdas(lms, 'test_data/shakespeare_sonnets.txt')
    probability = calculate_prob_with_backoff(wi,wn,lms,lambdas)
    #----------------------------------------------------
    sum_probability = sum_probability+probability

  perplex = math.exp(sum_probability/N)

  return perplex
  #preplex = sum_probability ** (1/N)
def calculate_prob_with_backoff(char, history, lms, lambdas):
  '''Uses interpolation to compute the probability of char given a series of 
     language models trained with different length n-grams.

   Inputs:
     char: Character to compute the probability of.
     history: A sequence of previous text.
     lms: A list of language models, outputted by calling train_char_lm.
     lambdas: A list of weights for each lambda model. These should sum to 1.
    
  Returns:
    Probability of char appearing next in the sequence.
  ''' 
  # TODO: YOUR CODE HRE
  order = len(lms)
  probability = 0
  while order > 0:
    hist = history[-order:]
    for n in range(len(lms)):
        if hist in lms[n]:
          for j in range(len(lms[n][hist])):
            if lms[n][hist][j][0] == char:
              probability = probability + lambdas[n]*math.log((1/lms[n][hist][j][1]))
    order-=1
  return probability


def set_lambdas(lms, dev_filename):
  '''Returns a list of lambda values that weight the contribution of each n-gram model

  This can either be done heuristically or by using a development set.

  Inputs:
    lms: A list of language models, outputted by calling train_char_lm.
    dev_filename: Path to a development text file to optionally use for tuning the lmabdas. 

  Returns:
    Probability of char appearing next in the sequence.
  '''
  # TODO: YOUR CODE HERE
  #dev = open(dev_fname,'r',encoding = 'utf8',errors='replace').read()
  #pad = "~" * order
  #dev = pad + dec
  #N = len(dev)-order

  lambdas = [1/len(lms)]*len(lms)
  #lambdas = [0.1,0.2,0.3,0.4]
  #lambdas = [0.4,0.3,0.2,0.1]
  #lambdas = [0.1,0.1,0.4,0.4]
  #lambdas = [0.4,0.4,0.1,0.1]
  #order = len(lms)
  #for i in range(len(lms)+1):
    #perplexity(dev_filename,lms[i],order=order)
  return lambdas
def classification(test_filename, order=1):
  '''Computes the perplexity of a text file given the language model.
  
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model.
  '''
  perplex = []
  countries = ['af','cn','de','fi','fr','in','ir','pk','za']
  lst = []
  probabilities = []
  k = 3
  number = 9
  for i in range(len(countries)):
    lms = []
    file_name = 'train/'+countries[i]+'.txt'
    for i in range(order):
      print(file_name)
      lms.append(train_char_lm(file_name, order=order, add_k=k))
    lst.append(lms)

  count = 0
  lines = ''

  lines = [line.rstrip('\n') for line in open(test_filename)]
  content = lines
  times = len(countries)
  #for i in range(len(lines)):
    #pad = "~" * order
    #lines[i] = pad+lines[i]
  lambdas = [0.0,0.5,0.5]
  for i in range(len(lines)):
    times = len(countries)
    probabilities = []
    for n in range(len(lst)):
      orders = order
      probability = 0
      while orders > 0:
        lines = content
        pad = "~" * order
        lines[i] = pad + lines[i]
        for j in range(len(lines[i])-order):
          wi = lines[i][j+order]
          wn = lines[i][j:j+order]
          if wn in lst[n][order-1]:
            for z in range(len(lst[n][order-1][wn])):
              if lst[n][order-1][wn][z][0] == wi:
               probability = probability + lambdas[order-1]*lst[n][order-1][wn][z][1]
        orders = orders - 1
      probabilities.append(probability)
    print(probabilities)
    index = np.argmax(probabilities)
    label = countries[index]
   
    with open('labels.txt','a') as out:
      out.write(label+'\n')
      count += 1
    #print(count)
      
    #index_max = np.argmax(probabilities)
    #print(index_max)
      #print(probabilities)
  
  #pad = "~"*order
  #for line in lines:
  # TODO: YOUR CODE HRER
    #line = pad + line
    #N = len(line)
    #sum_probability = 0
    #for i in range(N-order):
      #wi = line[i+order]
      #wn = line[i:i+order]
      #lambdas = set_lambdas(lms, 'test_data/shakespeare_sonnets.txt')
      #probability = calculate_prob_with_backoff(wi,wn,lms,lambdas)
      #sum_probability = sum_probability+probability

    #perplex.append(math.exp(sum_probability/N))

  return perplex
if __name__ == '__main__':
  print('Training language model')
  classification('test.txt', order = 3)
  #---------uncomment these for perplexity --------------
  #k = 1
  #order = 4
  #lms = []
  #while order >0:
    #lms.append(train_char_lm("shakespeare_input.txt", order=order, add_k=k))
    #order-=1
  #perplex = perplexity('test_data/nytimes_article.txt', lms, order=4)
  #perplex = perplexity('test_data/shakespeare_sonnets.txt', lms, order=4)
  #print(perplex)
  #------------------------------------------------------

  
