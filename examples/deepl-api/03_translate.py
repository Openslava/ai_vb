#
#

import io
import deepl
import os

env_auth_key = "DEEPL_AUTH_KEY"
env_server_url = "DEEPL_SERVER_URL"


def main() -> None:
    auth_key = os.getenv(env_auth_key)
    server_url = 'https://api-free.deepl.com' # os.getenv(env_server_url)
    file_name = './test.txt'
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
                source_lang="EN",
                target_lang="SK",
                filename=file_name,
                formality=deepl.Formality.DEFAULT,
        )
        doc_status.done

    print("Success")


if __name__ == "__main__":
    main()