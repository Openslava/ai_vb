#
#
# inicialisation 
# python -m venv openai-env
# source openai-env/bin/activate
# pip install --upgrade openai
# export OPENAI_API_KEY="..."
#

# open file 
import json
import argparse
from openai import OpenAI

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='translate into another language using OpensIA usage --source <filename.txt> --from <Romanian|Russian|English|...>  --to <Romanian|Russian|English|...> ')

# Define arguments with default values
parser.add_argument('--source',    type=str, default='./build/example.transcript.complete.txt', help='Source text file')
parser.add_argument('--from',  dest='from_arg',  type=str, default='English', help='translate from language')
parser.add_argument('--to',        type=str, default='Slovak', choices=['Slovak', 'English'], help='Translate into language')
parser.add_argument('--block_size',type=int, default=3000,  help='Chunk Block Size for translation')
parser.add_argument('--model',     type=str, default='gpt-4', help='Model to be used for translation')

# Parse the arguments
arguments = parser.parse_args()
arg_source = arguments.source
arg_from = arguments.from_arg
arg_to = arguments.to
arg_block_size = arguments.block_size
arg_model = arguments.model

client = OpenAI()
# "You are a helpful assistant that speach Romanian. Your task is to correct any spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."

# adjust 
system_prompt = "You are profesional translator from '{}' to {} language. Text that you shall translate was created by academic person that can describe complex thinks wiht symple words, use only the context provided".format(arg_from,arg_to)

def generate_translation(temperature, system_prompt, content):
    response = client.chat.completions.create(
        model=arg_model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )
    # usage = json.dumps(response.usage)
    # print (usage)
    return response.choices[0].message.content

corrected_text = ''
file_name = arg_source
chunk_count = 1
# using puthon load text file
with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
        # replace windows vs Linux CR or NL if mixed.
        line = line.replace('\n','')
        line = line.replace('\r','')
        if len(line) <=0 :
            continue 

        #split big paragraph
        if len(line) > (2*arg_block_size):
            sentences = line.split(',')
            #evaluate sentensed in line (paragraph)
            chunk = ''
            for sentence in sentences:
                if len(sentence) <=0 :
                    continue            
                if len(chunk + sentence) <= arg_block_size:
                    chunk += sentence + ","
                else:
                    print("---- chunk {} size {}...".format(chunk_count, len(chunk)))
                    corrected_text += generate_translation(0, system_prompt, chunk)
                    chunk = sentence + ","
                    chunk_count += 1
        else:
            # whole line is one chunk
            chunk = line

        if chunk:
            if len(chunk) > 0 :
                print("---- chunk {} size {}...".format(chunk_count, len(chunk)))
                corrected_text += generate_translation(0, system_prompt, chunk)
                chunk_count += 1

        corrected_text += "\n"

print("writing result size {}...\n".format(len(corrected_text)))
with open(file_name + '.translate.txt', 'w') as ftxt:
    ftxt.write(corrected_text)
