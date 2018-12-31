from arenaSimple_Any import runArena
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    runArena(LinearRegression(), "Linear", "CartPole-v1")