from sklearn.svm import SVR
from arenaSimple_Any import runArena

if __name__ == "__main__":
    runArena(SVR(), 'SVR', "CartPole-v1")
