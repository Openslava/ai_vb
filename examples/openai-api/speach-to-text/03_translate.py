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


from openai import OpenAI
client = OpenAI()
# "You are a helpful assistant that speach Romanian. Your task is to correct any spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."

# adjust 
system_prompt = "Translate the following Romanian text to Slovak: The text is spoken by female speaker."

def generate_translation(temperature, system_prompt, content):
    response = client.chat.completions.create(
        model="gpt-4",
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
block_size = int(3000)
file_name = "./build/test.transcript.complete.txt"
chunk_count = 1
# using puthon load text file
with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
        sentences = line.split(',')
        chunk = ''
        for sentence in sentences:
            if len(chunk + sentence) <= block_size:
                chunk += sentence + ","
            else:
                print("---- chunk {} size {}...".format(chunk_count, len(chunk)))
                corrected_text += generate_translation(0, system_prompt, chunk)
                chunk = sentence + ","
                chunk_count += 1
        if chunk:
            print("---- chunk {} size {}...".format(chunk_count, len(chunk)))
            corrected_text += generate_translation(0, system_prompt, chunk)
            chunk_count += 1

print("writing result size {}...\n".format(len(corrected_text)))
with open(file_name + '.translate.txt', 'w') as ftxt:
    ftxt.write(corrected_text)
