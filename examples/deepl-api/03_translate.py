#
#  pip install --upgrade deepl
#  export DEEPL_AUTH_KEY="..."
#  python ./03_translate.py  --source  ./12.txt --origin 'RU' --to 'SK'

import io
import deepl
import os
import argparse

env_auth_key = "DEEPL_AUTH_KEY"
env_server_url = "DEEPL_SERVER_URL"

parser = argparse.ArgumentParser(description='transcript audio file OpensIA usage --source <filename.mp3> --to <sk|en|ro|...>')

# Define arguments with default values
parser.add_argument('--source',    type=str, default='./test.txt', help='Source text file')
parser.add_argument('--origin',      type=str, default='EN', choices=['SK', 'EN','RU', 'RO'], help='Translate into language')
parser.add_argument('--to',        type=str, default='SK', choices=['SK', 'EN','RU', 'RO'], help='Translate into language')
arguments = parser.parse_args()
arg_source = arguments.source
arg_origin = arguments.origin
arg_to = arguments.to


def main() -> None:
    auth_key = os.getenv(env_auth_key)
    server_url = 'https://api-free.deepl.com' # os.getenv(env_server_url)
    file_name = arg_source
    if auth_key is None:
        raise Exception(
            f"Please provide authentication key via the {env_auth_key} "
            "environment variable or --auth_key argument"
        )

    # Create a Translator object, and call get_usage() to validate connection
    translator: deepl.Translator = deepl.Translator(
        auth_key, server_url=server_url
    )

    #ginfo: deepl.GlossaryInfo = translator.create_glossary(
    #    "Test Glossary", "EN", "SK", {"Yes": "Áno"}
    # )
    with open(file_name, "rb") as in_file, open(file_name + '.deepl.txt', "wb") as out_file:
        doc_status: deepl.DocumentStatus = translator.translate_document(
                in_file,
                out_file,
                source_lang=arg_origin,
                target_lang=arg_to,
                filename=file_name,
                formality=deepl.Formality.DEFAULT,
        )
        doc_status.done

    print("Success")


if __name__ == "__main__":
    main()