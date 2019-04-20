import random
import math
import numpy as np
from scipy import optimize as op

A = (100,90,80,70,60,50,40,30,20,10)  #A动作集
B = (100,90,80,70,60,50,40,30,20,10)  #B动作集

S = []  #状态集
Q = {}  #Q表
V = {}  #V表
Pai = {}  #π表

s0 = ()  #t=0时，初始状态s0

#每次迭代Q被更新的次数
Q_times = {}
pre_Q_times = {}

#保存每个Q更新前后的差值
Q_delta = {}
#对于每个s，保存π(s,all a)中更新前后差值最大的△π(s,a)的值
Pai_delta = {}

#每次迭代中，状态s出现的次数
s_times = {}
#每次迭代经历的t的个数
iteration_t = 1
#每次迭代中更新过的Q的前后差值
Q_delta_t = {}
Pai_delta_t = {}

#user j在t(用1,2,3...表示)时选择的provider为A(用1表示)或者B(用2表示),两者均不选(用0表示)
user_chooseProvider = {}

#t=0时，市场中的user数量
userNum_t0=10000;
#市场饱和时，user的总数量
userNum_stationary=10000;

#user j的marginal value
user_value={}

#t=0时A的运营成本
c0 = 5.0;

#users的初始需求
d0 = 0

#公式中的各个参数值
explor = 0.8
k = 0.07
#λ
gamma= 0.9
#Λ
lamb = 2
#β
beta = 0.01
#η
eta = 0.02


#读取文本文件
def readFile(fileName):
    dic = {}
    file = open(fileName)
    n = 0
    while True:
        stri = file.readline().strip()
        if not stri:
            break
        dic[n] = float(stri)
        n += 1
    file.close()
    return dic


def initialize():
    global S,Q,V,Pai,Q_times,pre_Q_times,Q_delta,Pai_delta,user_value

    user_value = readFile("value.txt")

    #初始化S
    for i in range(len(A)):
        for j in range(len(B)):
            state = (A[i],B[j])
            S.append(state)

    #初始化Q,V,Pai
    for i in range(len(S)):
        QA = {}
        PaiA = {}

        for j in range(len(A)):
            QB = {}

            for k in range(len(B)):
                QB[B[k]] = 1.0

            QA[A[j]] = QB
            PaiA[A[j]] = 1.0/len(A)

        Q[S[i]] = QA  #初始化Q
        V[S[i]] = 1.0  #初始化V
        Pai[S[i]] = PaiA  #初始化Pai

    #初始化Q_times
    for i in range(len(S)):
        dic1 = {}

        for j in range(len(A)):
            dic2 = {}

            for k in range(len(B)):
                dic2[B[k]] = 0

            dic1[A[j]] = dic2

        Q_times[S[i]] = dic1

    #初始化pre_Q_times,每次迭代开始时,pre_Q_times等于Q_times
    for i in range(len(S)):
        dic5 = {}

        for j in range(len(A)):
            dic6 = {}

            for k in range(len(B)):
                times = Q_times.get(S[i]).get(A[j]).get(B[k])
                dic6[B[k]] = times

            dic5[A[j]] = dic6

        pre_Q_times[S[i]] = dic5

    #初始化Q_delta
    for i in range(len(S)):
        dic3 = {}

        for j in range(len(A)):
            dic4 = {}

            for k in range(len(B)):
                #初始Q更新前后的差值为正无穷大
                dic4[B[k]] = float('inf')

            dic3[A[j]] = dic4

        Q_delta[S[i]] = dic3

    #初始化Pai_delta
    for i in range(len(S)):
        Pai_delta[S[i]] = float('inf')

def initialize_all():
    global S,Q_delta_t,Pai_delta_t,s_times,s0,d0,iteration_t,user_chooseProvider

    #初始化s_times
    for s in S:
        s_times[s] = 0

    #每次迭代开始，从S中随机选择一个s0
    s0 = S[random.randint(0,len(S))]
    d0 = 3.0
    #初始化每次迭代经历的t的个数
    iteration_t = 1

    #将t=0时存在于市场中的所有user选择的provider初始化为0
    for i in range(userNum_t0):
        user_chooseProvider[i] = 0

    #初始化Q_delta_t
    for i in range(len(S)):
        dic1 = {}

        for j in range(len(A)):
            dic2 = {}

            for k in range(len(B)):
                #初始Q更新前后的差值为负无穷大
                dic2[B[k]] = -float("inf")

            dic1[A[j]] = dic2

        Q_delta_t[S[i]] = dic1

    #初始化Pai_delta_t
    for i in range(len(S)):
        Pai_delta_t[S[i]] = -float("inf")

def takeAction(state):
    action = 0
    #产生一个0-1随机数r与explor比较
    r_explor = random.random()

    #如果小于explor，则均匀随机的选择一个action
    if r_explor <= explor:
        # 产生一个随机数，用来均匀随机选择一个action
        r_action = random.random()

        tar = 0
        for i in range(len(A)):
            if r_action >= tar/len(A) and r_action < (tar+1)/len(A):
                action = A[i]
                break
            tar+=1
    #如果不小于explor，则以概率分布π来选择一个动作
    else:
        r_action2 = random.random()

        tar = 0
        for i in range(len(A)):
            policy = tar + Pai.get(state).get(A[i])

            if r_action2 >= tar and r_action2 < policy:
                action = A[i]
                break

            tar = policy

    return action

#在S中找到和a与o对应的state
def findState(actionA,actionB):
    state = s0
    for i in range(len(S)):
        temp = S[i]
        if temp[0] == actionA and temp[1] == actionB:
            state = temp
    return state

#users选择provider，得到s'
def chooseProvider(state,t,actionA,actionB,t_priceA,t_priceB):
    user_chooseA = 0
    user_chooseB = 0
    user_free = 0

    #判断每个user应该选择哪个provider
    for j in range(userNum_stationary):
        #user j的marginal value
        value = user_value.get(j)

        #计算u_A,u_B,然后计算P(u_A)和p(u_B),再产生随机数,决定选择A还是B
		#分别计算user j选择A和B的value-price
        v_A = value - actionA
        v_B = value - actionB

        #若v_A和v_B均不大于0，则两者都不选
        if v_A <= 0 and v_B <= 0:
            user_free += 1
            user_chooseProvider[j] = 0
        elif v_A > 0 and v_B <= 0 :
            user_chooseA += 1
            user_chooseProvider[j] = 1
        elif v_A <= 0 and v_B > 0:
            user_chooseB += 1
            user_chooseProvider[j] = 2
        elif v_A > 0 and v_B > 0:
            #pChooseA = math.pow(math.e,v_A)/math.pow(math.e,v_A) + math.pow(math.e,v_B)
            pChooseA = 1 / (1 + math.exp((0.8 * (v_A - v_B) + 0.2 * (state[1] - state[0])) / (-1)))
            r = random.random()
            if r >= 0 and r < pChooseA:
                user_chooseA += 1
                user_chooseProvider[j] = 1
            if r >= pChooseA and r < 1:
                user_chooseB += 1
                user_chooseProvider[j] = 2

    #print("user的选择为:(" + str(user_chooseA) + "," + str(user_chooseB) + "," + str(user_free) + ")")

#计算t时A的运营成本cost
def cost(t,t_priceA,t_priceB):
    c = c0
    demand = demand_all(t,t_priceA,t_priceB)

    #如果A的总需求不为0
    if demand != 0:
        c = c0*math.pow(demand,-beta)*math.pow(math.e,-(eta*t))

    #print("运营成本=" + str(c0) )
    return c0

#计算t时user的需求(每个user的需求一样)
def demand(t,t_priceA,t_priceB):
    dem = d0
    """
    需扩展
    """
    return dem

#计算t时A收到的总需求
def demand_all(t,t_priceA,t_priceB):
    users_all = userNum_stationary
    demand_all = 0

    #定义t时选择A的user的数量
    user_chooseA = 0
    #遍历t时市场中的的每个user
    for j in range(users_all):
        if user_chooseProvider.get(j) == 1:
            #计算t时选择A的users的数量
            user_chooseA += 1

    demand_all = user_chooseA * demand(t,t_priceA,t_priceB)
    return demand_all

#计算t时A采取定价a的立即回报reward
def reward(t,actionA,t_priceA,t_priceB):
    c = cost(t,t_priceA,t_priceB)
    demand = demand_all(t,t_priceA,t_priceB)
    rew = demand * (actionA - c)
    return rew * 0.0001

#o'为o[j]时，计算v
def v_o(state,j):
    global Q,V,Pai

    solution = {}
    #当前s对应的π(s,a)
    pai_s = np.zeros(len(A))
    for i in range(len(pai_s)):
        pai_s[i] = Pai.get(state).get(A[i])

    #O[j]对应的求最小值时的公式的和的系数
    o_q = np.zeros((len(B),len(A)))
    for i in range(len(B)):
        for k in range(len(A)):
            o_q[i][k] = Q.get(state).get(A[k]).get(B[i])

    #定义、初始化目标函数系数矩阵
    x = np.zeros(len(A))
    for i in range(len(x)):
        x[i] = o_q[j][i]

    #定义约束条件系数矩阵a,b,c
    a = np.zeros((len(B)-1,len(A)))
    b = np.zeros((1,len(A)))
    a_e = np.zeros(9)

    #初始化a
    temp = 0
    for i in range(len(a)):
        if temp == j:
            temp += 1
        for k in range(len(a[i])):
            if k != len(a[i])-1:
                a[i][k] = o_q[j][k] - o_q[temp][k]
            else:
                a[i][k] = 0.0
        temp += 1

    #初始化b
    for i in range(len(b)):
        for k in range(len(b[i])):
            b[i][k] = 1.0
    b_e = np.array([1])

    #取值范围
    x1 = (0, 1)
    x2 = (0, 1)
    x3 = (0, 1)
    x4 = (0, 1)
    x5 = (0, 1)
    x6 = (0, 1)
    x7 = (0, 1)
    x8 = (0, 1)
    x9 = (0, 1)
    x10 = (0, 1)

    res = op.linprog(-x, a, a_e, b, b_e, bounds=(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10))
    sol = res.x

    #若有可行解，将线性规划求得的最小值赋值给v，并将其对应的一组xi赋值给数组pai_s
    v = -float("inf")
    if type(sol) == np.ndarray:
        v = 0
        for i in range(len(sol)):
            v += sol[i] * x[i]
        for i in range(len(pai_s)):
            pai_s[i] = sol[i]
        #当o'=O[j]时，若有可行解，输出v和Pai
        """
        stri = ""
        for i in range(len(pai_s)):
            stri = stri + str(pai_s[i]) + ","
        print("可行解v=" + str(v) + ",π(s)=(" + stri + ")\r\n")
        """
    else:
        # 若没有可行解，则定义v为负无穷大,其对应的pai_s为Pai表中根据当前s查出的一组值
        print("无可行解\r\n")

    solution['pai_s'] = pai_s
    solution['v'] = v

    return solution

def learn(state,actionA,actionB,t,t_priceA,t_priceB,alpha):
    global S,Q,V,Pai

    flag = 1  #用来指示该次learn过程对于每个actionB'是否都没有可行解
    s_next = s0

    s_next = findState(actionA,actionB)
    chooseProvider(state,t, actionA, actionB, t_priceA, t_priceB)

    #计算reward
    rew = reward(t,actionA,t_priceA,t_priceB)
    #print("reward=" + str(rew) + " ")

    #查询V(s')
    v_s_next = V.get(s_next)
    #print("V(s')=" + str(v_s_next) + " ")
    #print("Q=" + str(Q.get(state).get(actionA).get(actionB)) + " ")

    #更新Q(s, a, o)
    q = (1 - alpha) * (Q.get(state).get(actionA).get(actionB)) + alpha * (rew + gamma * v_s_next)
    Q.get(state).get(actionA)[actionB] = q

    #更新Pai
    prePai = np.zeros(len(A))
    for i in range(len(A)):
        prePai[i] = Pai.get(state).get(A[i])

    print("π(" + str(state[0]) + "," + str(state[1]) + ")=" + str(prePai) + ")")

    #定义learn后得到的o'在O中的下标
    index_o = 0
    solutionList = {}
    solution = {}
    for i in range(len(B)):
        solution = v_o(state,i)
        solutionList[i] = solution

    #找出最大solution，得到对应的v和π
    #若多个o得到的solution对应的v的值相等，则从中随机选择一个
    buffList = []
    maxSolution = solutionList[0]
    buffList.append(0)

    for j in range(len(solutionList)):
        if solutionList[j].get('v') > maxSolution.get('v'):
            maxSolution = solutionList[j]
            buffList.clear()
            buffList.append(j)
        elif solutionList[j].get('v') == maxSolution.get('v'):
            buffList.append(j)

    #从保存所有最大解的下标中随机选择一个(同时也是o'的下标)
    index_o = buffList[random.randint(0,len(buffList)-1)]
    buffList = None

    #/打印选择的最大解的对应的o
	#打印solutions[index_o]中的v值
    print("一组解中，最大的v值=" + str(solutionList[index_o].get('v')) + ",选择的o'=" + str(B[index_o]))

    #用solutions[index_o]中的π更新Pai(如果对所有o'都没有可行解，则π更新前后无变化)
    for j in range(len(A)):
        Pai.get(state)[A[j]] = solutionList[index_o].get('pai_s')[j]

    #更新V
	#当前s下对应的π(s,all)
    pai_s = np.zeros(len(A))
    for i in range(len(A)):
        pai_s[i] = Pai.get(state).get(A[i])
    #若solutions[index_o].v不为负无穷大，那么直接利用maxSolution.v值
    if solutionList[index_o].get('v') != -float("inf"):
        V[state] = solutionList[index_o].get('v')
    else:
        flag = 0

    return flag


#求数组的最大值
def maxArray(array):
    max = array[0]
    for i in range(len(array)):
        if array[i] > max :
            max = array[i]
    return max

#求有关Q值的表中最大值
def maxTableQ(dic):
    max = -float('inf')
    for keyS in dic:
        for keyA in dic.get(keyS):
            for keyB in dic.get(keyS).get(keyA):
                q = dic.get(keyS).get(keyA).get(keyB)
                if q>max:
                    max = q
    return max

#求有关π的表中的最大值
def maxTablePai(dic):
    max = -float('inf')
    for keyS in dic:
        pai = dic.get(keyS)
        if pai>max:
            max = pai
    return max

#更新pre_Q_times
def updatePreQTimes():
    global  pre_Q_times,Q_times

    for i in range(len(S)):
        for j in range(len(A)):
            for k in range(len(B)):
                temp = Q_times.get(S[i]).get(A[j]).get(B[k])
                pre_Q_times.get(S[i]).get(A[j])[B[k]] = temp

#文件操作
def appendFile(fileName,content):
    file = open(fileName,"a")
    file.write(content)
    file.close()

#从t=1~200进行一次迭代
def episode(path):
    global S,Q,V,Pai,Q_times,Q_delta_t,Q_delta,Pai_delta,Pai_delta_t,iteration_t

    state = ()
    state = s0
    t_priceA = s0[0]
    t_priceB = s0[1]

    #打印每次迭代时的s0
    #print("s0=" + "(" + str(s0[0]) + "," + str(s0[1]) + ")\r\n")
    t = 1

    timesA = np.zeros(len(A))
    timesB = np.zeros(len(A))
    maxQ = float("inf")
    maxPai = float("inf")

    #每个t后根据从1~t更新过的Q的更新前后的差值是否小于0.001来判断是否结束本次迭代
    while t<100000 or (maxQ>=0.01 and maxPai>=0.001) :
        #打印一次迭代中，每个t时的state
        #print("t="+str(t)+"时，s=("+str(state[0])+","+str(state[1])+")"+" user需求=" + str(demand(t,t_priceA,t_priceB))+"\r\n")

        n = 0
        while 1 :
            a = takeAction(state)
            b = B[random.randint(0,len(B)-1)]
            #print("a=" + str(a) + " b=" + str(b) + "\r\n")

            for i in range(len(A)):
                if a == A[i]:
                    timesA[i] += 1

            for i in range(len(B)):
                if b == B[i]:
                    timesB[i] += 1

            """
            for i in range(len(timesA)):
                print("A选中" + str(A[i]) + "的次数=" + str(timesA[i]) + " " + "B选中" + str(B[i]) + "的次数=" + str(timesB[i]))
            print("/n")
            """
            #alpha由Q当前Q(s,a,o)历史被更新的次数决定，包括当前即将更新的Q
            k = Q_times.get(state).get(a).get(b)
            Q_times.get(state).get(a)[b] = k+1
            alpha = 1.0/Q_times.get(state).get(a).get(b)
            #print("alpha=" + str(alpha) + " ")

            q1 = Q.get(state).get(a).get(b)
            #更新前对于s每个π(s,a)的值
            pai1 = np.zeros(len(A))
            for i in range(len(A)):
                pai1[i] = Pai.get(state).get(A[i])

            flag = learn(state,a,b,t,t_priceA,t_priceB,alpha)
            if flag == 1 :
                q2 = Q.get(state).get(a).get(b)
                deltaQ = abs(q2 - q1)
                Q_delta_t.get(state).get(a)[b] = deltaQ

                #将更新前后的Q的差值存入Q_delta
                Q_delta.get(state).get(a)[b] = deltaQ

                #更新后对于s每个π(s,a)的值与更新前对应的π(s,a)的差值
                deltaPai = np.zeros(len(A))
                for i in range(len(A)):
                    deltaPai[i] = Pai.get(state).get(A[i]) - pai1[i]

                #找出deltaPai中的最大值并插入Pai_delta中
                maxDeltaPai = maxArray(deltaPai)
                Pai_delta_t[state] = maxDeltaPai
                Pai_delta[state] = maxDeltaPai

                #print("△π(" + str(state[0]) + "," + str(state[1]) + ")=" + str(Pai_delta_t.get(state)) + "\r\n")
                break
            else:
                #若返回false,则对于每个o',由线性规划都得不到可行解，故回溯,去掉更新过的Q值以及alpha,接着循环
                n += 1
                #print("均没有可行解，回溯第" + str(n) + "次")
                Q.get(state).get(a)[b] = q1
                k1 = Q_times.get(state).get(a).get(b)
                Q_times.get(state).get(a)[b] = k1 - 1

        #当前t的价格用作下一个t时的t-1的价格
        t_priceA = a
        t_priceB = b
        #根据a和o'更新s
        new_state = findState(a,b)
        state = new_state

        if t % 50000 == 0:
            appendFile(path,"t="+str(t)+"时,Q,V,Pai表如下:\r\n")

            #Q表
            appendFile(path,"A的Q表如下:\r\n\r\n")
            for i in range(len(S)):
                for j in range(len(A)):
                    for k in range(len(B)):
                        appendFile(path,str(Q.get(S[i]).get(A[j]).get(B[k]))+"")
                    appendFile(path, "\r\n")
                appendFile(path, "s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"\r\n")

            #V表，π表
            appendFile(path, "A的V,Pai表如下\r\n\r\n")
            for i in range(len(S)):
                appendFile(path, "s(" + str(S[i][0]) + "," + str(S[i][1]) + ")" + " V:" + str(V.get(S[i])) + "\r\n")
                for j in range(len(A)):
                    appendFile(path, str(Pai.get(S[i]).get(A[j])) + " ")
                appendFile(path, "\r\n\r\n")

            #pre_Q_tiems
            appendFile(path, "pre_Q_times如下:\r\n\r\n")
            for i in range(len(S)):
                for j in range(len(A)):
                    for k in range(len(B)):
                        appendFile(path,str(pre_Q_times.get(S[i]).get(A[j]).get(B[k]))+"")
                    appendFile(path, "\r\n")
                appendFile(path, "s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"\r\n")

            #Q_times
            appendFile(path, "Q_times如下:\r\n\r\n")
            for i in range(len(S)):
                for j in range(len(A)):
                    for k in range(len(B)):
                        appendFile(path,str(Q_times.get(S[i]).get(A[j]).get(B[k]))+"")
                    appendFile(path, "\r\n")
                appendFile(path, "s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"\r\n")

            updatePreQTimes()  #更新pre_Q_times


            #打印Q_delta和Q_delta_t
            appendFile(path, "Q_delta_t如下:\r\n\r\n")
            for i in range(len(S)):
                for j in range(len(A)):
                    for k in range(len(B)):
                        appendFile(path,str(Q_delta_t.get(S[i]).get(A[j]).get(B[k]))+"")
                    appendFile(path, "\r\n")
                appendFile(path, "s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"\r\n")

            appendFile(path, "Q_delta如下:\r\n\r\n")
            for i in range(len(S)):
                for j in range(len(A)):
                    for k in range(len(B)):
                        appendFile(path,str(Q_delta.get(S[i]).get(A[j]).get(B[k]))+"")
                    appendFile(path, "\r\n")
                appendFile(path, "s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"\r\n")

            appendFile(path, "maxQ=" + str(maxTableQ(Q_delta_t)) + "\r\n\r\n")

            #打印Pai_delta和Pai_delta_t
            appendFile(path, "Pai_delta_t如下:\r\n\r\n")
            for i in range(len(S)):
                appendFile(path, str(Pai_delta_t.get(S[i])) + " ")
                if i>0 and i%9 ==0:
                    appendFile(path, "\r\n")

            appendFile(path, "Pai_delta如下:\r\n\r\n")
            for i in range(len(S)):
                appendFile(path, str(Pai_delta.get(S[i])) + " ")
                if i>0 and i%9 ==0:
                    appendFile(path, "\r\n")

            appendFile(path, "\r\nmaxPai=" + str(maxTablePai(Pai_delta_t)) + "\r\n\r\n")

        t += 1

        maxQ = maxTableQ(Q_delta_t)
        maxPai = maxTablePai(Pai_delta_t)
        #print("maxQ=" + str(maxQ) + " maxPai=" + str(maxPai) + "\r\n")

    """
    print("A的Q表如下:")
    for i in range(len(S)):
        for j in range(len(A)):
            for k in range(len(B)):
                q = Q.get(S[i]).get(A[j]).get(B[k])
                print(str(q) + ' ')
            print("/n")
        print("s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"/n")

    print("/nQ_delta_t如下:")
    for i in range(len(S)):
        for j in range(len(A)):
            for k in range(len(B)):
                q = Q_delta_t.get(S[i]).get(A[j]).get(B[k])
                print(str(q) + ' ')
            print("/n")
        print("s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"/n")

    print("/nQ_delta如下:")
    for i in range(len(S)):
        for j in range(len(A)):
            for k in range(len(B)):
                q = Q_delta.get(S[i]).get(A[j]).get(B[k])
                print(str(q) + ' ')
            print("/n")
        print("s(" + str(S[i][0]) + "," + str(S[i][1]) + ")"+"/n")

    """
    #每次迭代后清空Q_delta_iteration
    Q_delta_t.clear()
    Pai_delta_t.clear()

    iteration_t = t - 1

    max_deltaQ = maxTableQ(Q_delta)
    max_deltaPai = maxTablePai(Pai_delta)
    max = [max_deltaQ,max_deltaPai]

    return max



