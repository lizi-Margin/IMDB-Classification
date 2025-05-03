import numpy as np
from database_wrapper import get_IMDB
from cls import classify
from eval import eval

def main():
    X_train, X_test, y_train, y_test = get_IMDB()
    result = classify(X_train, X_test, y_train, y_test)
    result = eval(result, y_test)

    # pretty print result dict
    for key, value in result.items():
        if (isinstance(value, list) or isinstance(value, np.ndarray)) and len(value) > 10:
            print(f"{key}: {value[:10]}... len={len(value)}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
