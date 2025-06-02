import sys
from src.logger import logging

def error_message_detail(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_details)
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message