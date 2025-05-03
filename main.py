from database_wrapper import get_IMDB
from cls import classify
from eval import eval

def main():
    X_train, X_test, y_train, y_test = get_IMDB()
    result = classify(X_train, X_test, y_train, y_test)
    result = eval(result, y_test)

    # pretty print result dict
    for key, value in result.items(): print(f"{key}: {value}")

if __name__ == "__main__":
    main()
