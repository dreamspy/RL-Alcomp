from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from arenaSimple_Any import runArena

if __name__ == "__main__":
    runArena(MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1)), 'GBDT', 'CartPole-v1')
