import sys
import traceback
from typing import Optional, cast


class DocumentPortalException(Exception):
    def __init__(self, error_message, error_details: Optional[object] = None):
        # Normalize the error message into a string (whether it's an Exception or text)
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # Initialize variables to hold exception info
        exc_type = exc_value = exc_tb = None

        # Case 1: If no specific error details are passed, capture the current exception info
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            # Case 2: If the details object has an exc_info() method (like sys)
            if hasattr(error_details, "exc_info"):
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()

            # Case 3: If error_details is itself an exception instance
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__

            # Case 4: Fallback to the current context’s exception info
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk through traceback to find the last (most relevant) frame
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # Extract file name and line number where the error occurred
        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Generate a full traceback string for logging or debugging
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        # Initialize the base Exception class with this formatted error string
        super().__init__(self.__str__())

    def __str__(self):
        # Create a readable string summarizing file, line, and message
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"

        # Append full traceback if available
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        # Developer-friendly representation (useful in debugging)
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"
