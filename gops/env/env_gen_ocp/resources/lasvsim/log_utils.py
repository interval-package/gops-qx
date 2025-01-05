import time
import grpc
import functools


class LoggingInterceptor(grpc.UnaryUnaryClientInterceptor):
    def intercept_unary_unary(self, continuation, client_call_details, request):
        start_time = time.time()
        response = continuation(client_call_details, request)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        if elapsed_time > 10:  # 10 milliseconds
            pass
            # print(f"RPC call to {client_call_details.method} took {elapsed_time:.2f} ms")
        return response

def timeit(func):
    """Decorator to measure the execution time of a method."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()  # Start the timer
        value = func(*args, **kwargs)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        # print(f"Function {func.__name__!r} took {elapsed_time:.4f} seconds to complete.")
        return value

    return wrapper_timer