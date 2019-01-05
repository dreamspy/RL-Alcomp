from arenaSimple_Any import runArena
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    runArena(MultiOutputRegressor(LinearRegression()), "Linear", "CartPole-v1")