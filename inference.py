from re import L


from model import *


class Inference:
    def __init__(self,path):
        self.model = self.load_model(path)


    def load_model(self,path):

        print(no_layers,vocab_size,hidden_dim,embedding_dim)

        model = Model_predict(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
        model.load_state_dict(torch.load(path, map_location = torch.device("cpu")))
        model.eval()

        return model

    def padding_(self,sentences, seq_len):
        features = np.zeros((len(sentences), seq_len),dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features

    def infer(self,text):
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(self.padding_(word_seq,25))
        inputs = pad.to(device)
        batch_size = 1
        h = self.model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = self.model(inputs, h)
        pro = output.item()
        status = "positive" if pro > 0.5 else "negative"
        return status

    



# infer = Inference("/home/manas/Desktop/Job Assessments/truefoundry/sa.pth")
# a = infer.infer("The best person on the planet, really helpful, great!")
# print(a)
