from contextlib import contextmanager
import os
import sys

@contextmanager
def suppress_c_stdout():
    # Flush Python's print buffer first
    sys.stdout.flush()
    # The OS-level file descriptor for stdout is always 1
    stdout_fd = 1
    # Duplicate the original stdout so we can restore it later
    old_stdout_fd = os.dup(stdout_fd)
    # Open the system's null device (the "black hole")
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        # Override standard output with the null device
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        # Restore original standard output
        os.dup2(old_stdout_fd, stdout_fd)
        # Clean up our temporary file descriptors
        os.close(old_stdout_fd)
        os.close(devnull_fd)