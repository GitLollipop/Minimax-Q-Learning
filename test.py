import MinimaxQ as minimax
import datetime
import time

#新建一个文本文件，用于保存实验结果 (路径)
fileName="result/MR_stationary.txt"

#初始化
minimax.initialize()

#迭代
iterations = 0
max_delta = [1.0,1.0]
while max_delta[0] >= 0.01 and max_delta[1] >= 0.001:
    print("第" + str(iterations + 1) + "次迭代，")
    startTime = datetime.datetime.now()

    minimax.initialize_all()

    stri = "result/MR_StationaryTable" + str(iterations + 1) + ".log"
    max_delta = minimax.episode(stri)
    print("经历t的个数为:" + str(minimax.iteration_t))

    endTime = datetime.datetime.now()
    print("共需要时间：" + str((endTime - startTime).microseconds) + "ms  max_deltaQ的值为：" + str(max_delta[0]) + "  max_deltaPai的值为：" + str(max_delta[1]) + "\r\n\r\n\r\n")

    iterations += 1

    #每次迭代后在新建的文本文件中追加V,Pai,Q
    minimax.appendFile(fileName, "第" + str(iterations) + "次迭代，经历t的个数为:" + str(minimax.iteration_t) + "\r\n\r\n")
    for i in range(len(minimax.S)):
        #V
        minimax.appendFile(fileName, "s(" + str(minimax.S[i][0]) + "," + str(minimax.S[i][1]) + ")" + " V:" + str(minimax.V.get(minimax.S[i])) + "\r\n")
        #Pai
        for j in range(len(minimax.A)):
            minimax.appendFile(fileName, str(minimax.Pai.get(minimax.S[i]).get(minimax.A[j])) + " ")
        minimax.appendFile(fileName, "\r\n\r\n")

    #Q
    for i in range(len(minimax.S)):
        for j in range(len(minimax.A)):
            for k in range(len(minimax.B)):
                minimax.appendFile(fileName, str(minimax.Q.get(minimax.S[i]).get(minimax.A[j]).get(minimax.B[k])) + " ")
            minimax.appendFile(fileName,"\r\n")
        minimax.appendFile(fileName,"s(" + str(minimax.S[i][0]) + "," + str(minimax.S[i][1]) + ")"+"\r\n\r\n")
    minimax.appendFile(fileName,"共需要时间：" + str((endTime - startTime).microseconds) + "ms  max_deltaQ的值为：" + str(max_delta[0]) + "  max_deltaPai的值为：" + str(max_delta[1]) + "\r\n\r\n\r\n")

print("总迭代次数为:" + str(iterations))




