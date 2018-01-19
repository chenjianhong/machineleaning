#coding: utf-8



class HMM(object):
    '''
    参考:
    https://github.com/jason2506/PythonHMM/blob/master/hmm.py
    http://blog.csdn.net/wds2006sdo/article/details/75212599
    '''

    def __init__(self,hidden_states,output_states,start_prob,transfer_prob,emit_prob):
        self.hidden_states = hidden_states
        self.output_states = output_states
        self.start_prob = start_prob
        self.transfer_prob = transfer_prob
        self.emit_prob = emit_prob

    def __forward__(self,sequence):
        prob_seq_prob = []
        for i in range(1,len(sequence)):
            s = sequence[i]
            prob_seq_prob.append(dict())
            for end in self.hidden_states:
                for start in self.hidden_states:
                    prob_seq_prob[i][end] += prob_seq_prob[i-1][start] * self.transfer_prob[start][end] if i > 1 else self.start_prob[start] * self.transfer_prob[start][end]
                prob_seq_prob[i][end] = prob_seq_prob[i][end] * self.emit_prob[end][s]
        return prob_seq_prob

    def evaluate(self,sequence):
        prob = 0
        if len(sequence)==0:
            return prob
        prob_list = self.__format__(sequence)
        return prob_list[-1][sequence[-1]]
