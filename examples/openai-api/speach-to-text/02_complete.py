#
#
# inicialisation 
# python -m venv openai-env
# source openai-env/bin/activate
# pip install --upgrade openai
# export OPENAI_API_KEY="..."
#

# open file 

from openai import OpenAI
client = OpenAI()
# "You are a helpful assistant that speach Romanian. Your task is to correct any spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."

# adjust 
system_prompt = "You are a helpful assistant that speach Romanian. Your task is to correct any spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. The text is spoken by female speaker."

def generate_corrected_transcript(temperature, system_prompt, content):
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
    return response.choices[0].message.content

corrected_text = ''
block_size = int(3000)
chunk_count = 1
file_name = "./build/test.transcipt.txt"
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
                corrected_text += generate_corrected_transcript(0, system_prompt, chunk)
                chunk = sentence + ","
                chunk_count += 1                
        if chunk:
            print("---- chunk {} size {}...".format(chunk_count, len(chunk)))
            corrected_text += generate_corrected_transcript(0, system_prompt, chunk)
            chunk_count += 1

print('writing result...\n')
with open(file_name + '.cpmplete.txt', 'w') as ftxt:
    ftxt.write(corrected_text)
