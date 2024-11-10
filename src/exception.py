import sys
import traceback
import logging
from src import my_logging

def error(err:sys):
    exc_type, exc_value, exc_tb= sys.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    func_name = exc_tb.tb_frame.f_code.co_name
    msg=f"you have a {exc_type}error\n{filename}\n in line:{lineno} function:{func_name}"
    return msg


class CustomException(Exception):
    def __init__(self, err, sys):
        self.error_msg= error(err)
        super().__init__(self.error_msg)
        

    def __srt__(self):
        return self.error_msg
    
