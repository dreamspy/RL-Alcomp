
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from arenaSimple_Any import runArena

if __name__ == "__main__":
    runArena((GridSearchCV(estimator=(Pipeline([('scl', StandardScaler()),
                          ('reg', SVR())])),
                           param_grid={
        'reg__C': [0.01, 0.1, 1, 10],
        'reg__epsilon': [0.1, 0.2, 0.3],
        'degree': [10,10,10]
    },
                           cv=10,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)), 'Grid_SVR', 'CartPole-v1')
