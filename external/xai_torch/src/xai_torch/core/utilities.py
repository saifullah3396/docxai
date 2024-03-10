def add_task(tasks):
    def decorator(function):
        tasks[function.__name__] = function
        return function

    return decorator
