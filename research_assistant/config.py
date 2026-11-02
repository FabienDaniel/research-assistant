import logging
import os
import warnings

from dotenv import find_dotenv, load_dotenv

warnings.filterwarnings("ignore")

ORACLE_DSN = "127.0.0.1:1521/FREEPDB1"
ORACLE_PROGRAM = "devrel.deeplearning.course_1"
VECTOR_USER = "VECTOR"
VECTOR_PASSWORD = "VectorPwd_2025"


# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService
def load_env():
    _ = load_dotenv(find_dotenv())


def suppress_warnings():
    # HuggingFace / transformers log messages
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key
