from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from arenaSimple_Any import runArena

if __name__ == "__main__":
    runArena(MultiOutputRegressor(SVR(gamma='auto')), 'SVR', "CartPole-v1")
