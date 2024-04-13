import numpy as np
from collections import Counter

class child:
    def __init__(s, attribute=None, th=None, x_lft=None, x_rt=None, *, ans=None):
        s.attribute = attribute
        s.th = th
        s.x_lft = x_lft
        s.x_rt = x_rt
        s.ans = ans

    def child_node(s):
        return s.ans is not None
    
class _decision_tree_:
    def __init__(s, ms_split=2, d1=5, nf=None):
        s.ms_split = ms_split
        s.d1 = d1
        s.nf = nf
        s.root = None

    # method to predict the outcome of provided data
    def _pred_(s, k):
        return np.array([s.tree_tr(k, s.root) for k in k])

    # method to train the data
    def trn(s, k, l):
        s.nf = np.shape(k)[1] if not s.nf else min(s.nf, np.shape(k)[1])
        s.root = s._tree_increment_(k, l)
    
    # method for increasing height of the tree till it reaches a stopping point
    def _tree_increment_(s, k, l, d1=0):
        test_n, attribute_n = np.shape(k)
        flags_n = len(np.unique(l))
         
        # picking attributes at random to reach a conclusion
        index_ft = np.random.choice(attribute_n, s.nf, replace=False)

        #choose the optimum split based on information gn with greedy method
        bf, bt = s.good_cr(k, l, index_ft)

        #criteria for stopping
        if (d1 >= s.d1
                or flags_n == 1
                or test_n < s.ms_split):
            l_val = s.lbl_cmn(l)
            return child(ans=l_val)
        
        #icrementing the child nodes that result from the split operation
        index_l, index_r = s.diff(k[:, bf], bt)
        x_lft = s._tree_increment_(k[index_l, :], l[index_l], d1+1)
        x_rt = s._tree_increment_(k[index_r, :], l[index_r], d1+1)
        return child(bf, bt, x_lft, x_rt)
    
    # method to choose the ideal attribute element for data splitting
    def good_cr(s, k, l, index_ft):
        bg = -1
        index_spl, th_spl = None, None
        for index_feat in index_ft:
            clmn = k[:, index_feat]
            tholds = np.unique(clmn)
            for th in tholds:
                gn = s.gn_info(l, clmn, th)

                if gn > bg:
                    bg = gn
                    index_spl = index_feat
                    th_spl = th

        return index_spl, th_spl

    def gn_info(s, l, clmn, th_spl):
        #root-child node entropy
        ent_root = ent(l)

        #split generate
        index_l, index_r = s.diff(clmn, th_spl)

        if len(index_l) == 0 or len(index_r) == 0:
            return 0

        #calculate the average of the nodes entropy
        a = len(l)
        left_a, right_a = len(index_l), len(index_r)
        left_ent, right_ent = ent(l[index_l]), ent(l[index_r])
        ent_new = (left_a / a) * left_ent + (right_a / a) * right_ent

        #difference of loss between before and after split is information gain
        _gn_info_= ent_root - ent_new
        return _gn_info_

    # tree traversal to compute the result of data
    def tree_tr(s, k, child):
        if child.child_node():
            return child.ans

        if k[child.attribute] <= child.th:
            return s.tree_tr(k, child.x_lft)
        return s.tree_tr(k, child.x_rt)

    def lbl_cmn(s, l):
        flag1 = Counter(l)
        cmn_highest = flag1.most_common(1)[0][0]
        return cmn_highest


    # split the data
    def diff(s, clmn, th_spl):
        index_l = np.argwhere(clmn <= th_spl).flatten()
        index_r = np.argwhere(clmn > th_spl).flatten()
        return index_l, index_r
 
# Calculating the _acc_ 
def _acc_(q_pred,q_act):
    flag_2=0
    for(q_pd,q_actual) in zip(q_pred,q_act):
        if(q_pd==q_actual):
           flag_2+=1
    return flag_2/len(q_pred)


# compute the entropy
def ent(l):
    dict = np.bincount(l)
    res = dict / len(l)
    return -np.sum([q * np.log2(q) for q in res if q > 0])   

def main():
    print('START Q1_AB\a')  

    train_data = np.array([[1.5963600450124, 75.717194178189, 23],
           [1.6990610819676, 83.477307503684, 25],
           [1.5052092436, 74.642420817737,    21],
           [1.5738635789008, 78.562465284603, 30],
           [1.796178772769, 74.566117057707,  29],
           [1.6274618774347, 82.250591567161, 21],
           [1.6396843250708, 71.37567170848,  20],
           [1.538505823668, 77.418902097029,  32],
           [1.6488692005889, 76.333044488477, 26],
           [1.7233804613095, 85.812112126306, 27],
           [1.7389100516771, 76.424421782215, 24],
           [1.5775696242624, 77.201404139171, 29],
           [1.7359417237856, 77.004988515324, 20],
           [1.5510482441354, 72.950756316157, 24],
           [1.5765653263667, 74.750113664457, 34],
           [1.4916026885377, 65.880438515643, 28],
           [1.6755053770068, 78.901754249459, 22],
           [1.4805881225567, 69.652364469244, 30],
           [1.6343943760912, 73.998278712613, 30],
           [1.6338449829543, 79.216500811112, 27],
           [1.5014451222259, 66.917339299419, 27],
           [1.8575887178701, 79.942454850988, 28],
           [1.6805940669394, 78.213519314007, 27],
           [1.6888905106948, 83.031099742808, 20],
           [1.7055120272359, 84.233282531303, 18],
           [1.5681965896812, 74.753880204215, 22],
           [1.6857758389206, 84.014217544019, 25],
           [1.7767370337678, 75.709336556562, 27],
           [1.6760125952287, 74.034126149139, 28],
           [1.5999112612548, 72.040030344184, 27],
           [1.6770845322305, 76.149431872551, 25],
           [1.7596128136991, 87.366395298795, 29],
           [1.5344541456027, 73.832214971449, 22],
           [1.5992629534387, 82.4806916967,   34],
           [1.6714162787917, 67.986534194515, 29],
           [1.7070831676329, 78.269583353177, 25],
           [1.5691295338456, 81.09431696972,  27],
           [1.7767893419281, 76.910413184648, 30],
           [1.5448153215763, 76.888087599642, 32],
           [1.5452842691008, 69.761889289463, 30],
           [1.6469991919639, 82.289126983444, 18],
           [1.6353732734723, 77.829257585654, 19],
           [1.7175342426502, 85.002276406574, 26],
           [1.6163551692382, 77.247935733799, 21],
           [1.6876845881843, 85.616829192322, 27],
           [1.5472705508274, 64.474350365634, 23],
           [1.558229415357, 80.382011318379,  21],
           [1.6242189230632, 69.567339939973, 28],
           [1.8215645865237, 78.163631826626, 22],
           [1.6984142478298, 69.884030497097, 26]])
    # 0 will represent W(Women) and 1 will represent M(Men)
    tr_y=np.array([0,1,0,1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1])

    tst_x = [[1.6468551415123, 82.666468220128, 29],
          [1.5727791290292, 75.545348033094, 24],
          [1.8086593470477, 78.093913654921, 27],
          [1.613966988578, 76.083586505149,  23],
          [1.6603990297076, 70.539053122611, 24],
          [1.6737443242383, 66.042005829182, 28],
          [1.6824912337281, 81.061984274536, 29],
          [1.5301691510101, 77.26547501308,  22],
          [1.7392340943261, 92.752488433153, 24],
          [1.6427105169884, 83.322790265985, 30],
          [1.5889040551166, 74.848224733663, 25],
          [1.5051718284868, 80.078271153645, 31],
          [1.729420786579, 81.936423109142,  26],
          [1.7352568354092, 85.497712687992, 19],
          [1.5056950011245, 73.726557750383, 24],
          [1.772404089054, 75.534265951718,  30],
          [1.5212346939173, 74.355845722315, 29],
          [1.8184515409355, 85.705767969326, 25],
          [1.7307897479464, 84.277029918205, 28],
          [1.6372690389158, 72.289040612489, 27],
          [1.6856953072545, 70.406532419182, 28],
          [1.832494802635, 81.627925524191,  27],
          [1.5061197864796, 85.886760677468, 31],
          [1.5970906671458, 71.755566818152, 27],
          [1.6780459059283, 78.900587239209, 25],
          [1.6356901170146, 84.066566323977, 21],
          [1.6085494116591, 70.950456539016, 30],
          [1.5873479102442, 77.558144903338, 25],
          [1.7542078120838, 75.3117550236,   26],
          [1.642417315747, 67.97377818999,   31],
          [1.5744266340913, 81.767568318602, 23],
          [1.8470601407979, 68.606183538532, 30],
          [1.7119387468283, 80.560922353487, 27],
          [1.6169930563306, 75.538611935125, 27],
          [1.6355653058986, 78.49626023408,  24],
          [1.6035395957618, 79.226052358485, 33],
          [1.662787957279, 76.865925681154,  25],
          [1.5889291137091, 76.548543553914, 28],
          [1.9058127964477, 82.56539915922,  25],
          [1.694633493614, 62.870480634419,  21],
          [1.7635692396034, 82.479783004684, 27],
          [1.6645292231449, 75.838104636904, 29],
          [1.7201968406129, 81.134689293557, 24],
          [1.5775563651749, 65.920103519266, 24],
          [1.6521294216004, 83.312640709417, 28],
          [1.5597501915973, 76.475667826389, 30],
          [1.7847561120027, 83.363676219109, 29],
          [1.6765690500715, 73.98959022721,  23],
          [1.6749260607992, 73.687015573315, 27],
          [1.58582362825, 71.713707691505,   28],
          [1.5893375739649, 74.248033504548, 27],
          [1.6084440045081, 71.126430164213, 27],
          [1.6048804804343, 82.049319162211, 26],
          [1.5774196609804, 70.878214496062, 24],
          [1.6799586185525, 75.649534976838, 29],
          [1.7315642636281, 92.12183674186,  29],
          [1.5563282000349, 69.312673560451, 32],
          [1.7784349641893, 83.464562543,    26],
          [1.7270244609765, 76.599791001341, 22],
          [1.6372540837311, 74.746741127229, 30],
          [1.582550559056, 73.440027907722,  23],
          [1.722864383186, 79.37821152354,   20],
          [1.5247544081009, 70.601290492141, 27],
          [1.580858666774, 70.146982323579,  24],
          [1.703343390074, 90.153276095421,  22],
          [1.5339948635367, 59.675627532338, 25],
          [1.8095306490733, 86.001187990639, 20],
          [1.7454786971676, 85.212429336602, 22],
          [1.6343303342105, 85.46378358014,  32],
          [1.5983479173071, 79.323905480504, 27]]
  
    tst_y=[1,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,0]

    for d1 in range(1,6):
        # construction method called
        tree_dt = _decision_tree_(d1=d1)
        # fit the data in decision tree
        tree_dt.trn(train_data, tr_y)

        # prediction of given test data 
        prediction_test_data = tree_dt._pred_(tst_x)
        
        # prediction of given train data 
        prediction_train_data= tree_dt._pred_(train_data)
        
        res_accuracy_test=_acc_(prediction_test_data,tst_y)
        res_accuracy_train=_acc_(prediction_train_data,tr_y)

        print(" Depth number ", d1)
        print("Accuracy of ","|","Train = ",res_accuracy_train,"|","Test = ",round(res_accuracy_test,2))

    
    print('END Q1_AB\a')

if __name__ == "__main__":
    main()

''' Question :  For which depths does the result indicate overfitting?
    Answer :  although testing accuracy is irregular, from depth number 3 training accuracy is increasing.
            Therefore, it is displaying overfitting from depth number 3.'''