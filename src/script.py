import math

################# classes for models
class Model:
    
    def __init__(self):
        self.voc = {}
        self.cnt = 0
        
    def build_voc(self, data):
        for i in range(len(data)):
            sentence = data[i]
            for j in range(len(sentence)):
                if sentence[j] in self.voc:
                    self.voc[sentence[j]] += 1
                else:
                    self.voc[sentence[j]] = 1
                self.cnt += 1   
     
    def get_occ(self, word):   
        if word in self.voc:
            return self.voc[word]
        else:
            return 0
        
class ZerogramModel(Model):
    
    def fit(self, data):
        self.build_voc(data)
        
    def get_prob(self, word):
        if word in self.voc:
            return 1 / self.cnt
        else:
            return 0
        
    def predict(self, data):
        counter = 0
        tmp = 0
        for i in range(len(data)):
            sentence = data[i]
            for j in  range(len(sentence)):
                prob = self.get_prob(sentence[j])
                if prob == 0:
                    tmp += -math.inf
                else:
                    tmp += math.log(prob) / math.log(2) 
                counter += 1
        # entropy = prob of each token / number of tokens
        entropy = -1 / counter * tmp
        print ('Zerogram model entropy is %f' % entropy)
        perplexity = math.pow(2, entropy)
        print ('Zerogram model perplexity is %f' % perplexity)
        
class UnigramModel(Model):
    
    def fit(self, data):
        self.build_voc(data)
        
    def get_prob(self, word):
        if word in self.voc:
            return self.voc[word] / self.cnt
        else:
            return 0    
        
    def predict(self, data):
        counter = 0
        tmp = 0
        for i in range(len(data)):
            sentence = data[i]
            for j in  range(len(sentence)):
                prob = self.get_prob(sentence[j])
                if prob == 0:
                    tmp += -math.inf
                else:
                    tmp += math.log(prob) / math.log(2) 
                counter += 1
        # entropy = prob of each token / number of tokens
        entropy = -1 / counter * tmp
        print ('Unigram model entropy is %f' % entropy)
        perplexity = math.pow(2, entropy)
        print ('Unigram model perplexity is %f' % perplexity)
        
class BigramModel(Model):
    
    def __init__(self):
        self.voc = {}
        self.bi_voc = {}
        self.cnt = 0
        
    def build_bigram_voc(self, data):
        for i in range(len(data)):
            sentence = data[i]
            for j in range(len(sentence) - 1):
                temp = sentence[j] + CONCAT_SYM + sentence[j + 1]
                if temp in self.bi_voc:
                    self.bi_voc[temp] += 1
                else:
                    self.bi_voc[temp] = 1
        
    def fit(self, data):
        self.build_voc(data)
        self.build_bigram_voc(data)
        
    def get_prob(self, word1, word2):
        temp = word1 + CONCAT_SYM + word2
        if temp in self.bi_voc and word1 in self.voc:
            return self.bi_voc[temp] / self.voc[word1]
        else:
            return 0    
     
    def predict(self, data):
        counter = 0
        tmp = 0
        for i in range(len(data)):
            sentence = data[i]
            for j in range(len(sentence) - 1):
                prob = self.get_prob(sentence[j], sentence[j + 1])
                if prob == 0:
                    tmp += -math.inf
                else:
                    tmp += math.log(prob,2) 
                counter += 1
        # entropy = prob of each token / number of tokens
        entropy = -1 / counter * tmp
        print ('Bigram model entropy is %f' % entropy)
        # perplexity 2^entropy
        perplexity = math.pow(2, entropy)
        print ('Bigram model perplexity is %f' % perplexity)
        
################# functions

def create_bigrams(data):
    bigrams = []
    for i in range(len(data)):
        sentence = data[i]
        for j in range(len(sentence) - 1):
            bigrams = sentence[j] + CONCAT_SYM + sentence[j + 1]
    return bigrams
                
# loads txt file
def load_file(filename):
    sentences = []
    # reads file line by line and mark start each sentence with <s> and end
    # with </s>
    with open(filename, encoding="utf8") as content:
        data = content.readlines()
        # remove empty chars like '/n'
        data = [x.strip() for x in data] 
        # go through all sentences in datafiles
        for index in range(len(data)):
            # load sentence by sentece
            sentence = data[index]
            # tokenize sentence
            temp = sentence.split(' ')
            # insert <s> tag like start of sentence
            temp.insert(0, "<s>")
            # insert </s> tag like end of sentence
            temp.append('</s>')
            sentences.append(temp)
        
    print('### Sucessfully loaded file \'' + filename + '\' with ' 
          + str(len(sentences)) + '\' items')
    return sentences


################# main

# constants 
TRAIN_DATA_PATH = '../data/train.txt'
TEST_DATA_PATH = '../data/test.txt'
HELDOUT_DATA_PATH = '../data/heldout.txt'
CONCAT_SYM = '###'

print ('#### Loading data started')
train_data = load_file(TRAIN_DATA_PATH)
heldout_data = load_file(HELDOUT_DATA_PATH)
test_data = load_file(TEST_DATA_PATH)
print ('#### Loading data done')
       
# zerogram model       
zerogram_model = ZerogramModel()
zerogram_model.fit(train_data)
prob = zerogram_model.get_prob('Robert')
print('The probability of word \'Robert\' in Zerogram language model is %f' %prob)

# unigram model       
unigram_model = UnigramModel()
unigram_model.fit(train_data)
prob = unigram_model.get_prob('Robert')
print('The probability of word \'Robert\' in Unigram language model is %f' %prob)

# bigram model       
bigram_model = BigramModel()
bigram_model.fit(train_data)
prob = bigram_model.get_prob('Robert', 'Lang')
print('The probability of words \'Robert Lang\' in Bigram language model is %f' %prob)

# test_bigrams = create_bigrams(test_data) - the entropy and perplexity will be
# Infinity if we will find unseen sample according to equation for entropy
zerogram_model.predict(test_data)
unigram_model.predict(test_data)
bigram_model.predict(test_data)

# next we will use smoothing for the posibility evaluate unseen samples