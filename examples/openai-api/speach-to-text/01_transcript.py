#
## use appropriate shell e.g. bash in WSL on Windows
## inicialisation 
#  python -m venv openai-env
#  source openai-env/bin/activate
#  pip install --upgrade openai
#  export OPENAI_API_KEY="..."
#  python ./01_transcript.py  --source  /mnt/g/repo/github/ai/ai_vb/examples/openai-api/speach-to-text/build/w2.mp3 

# convert audio to text


import argparse
from openai import OpenAI

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='transcript audio file OpensIA usage --source <filename.mp3> --to <sk|en|ro|...>')

# Define arguments with default values
parser.add_argument('--source',    type=str, default='./build/example.transcript.complete.txt', help='Source text file')
parser.add_argument('--to',        type=str, default='sk', choices=['sk', 'en','ro'], help='Translate into language')
parser.add_argument('--model',     type=str, default='whisper-1', help='Model to be used for translation')
arguments = parser.parse_args()
arg_source = arguments.source
arg_to = arguments.to
arg_model = arguments.model

client = OpenAI()

# rumenian prompt
# prompt = "Ești un asistent de ajutor care vorbește românește. Sarcina dumneavoastră este să corectați orice discrepanțe de ortografie din textul transcris. Adăugați doar semnele de punctuație necesare, cum ar fi puncte, virgule și majuscule și utilizați numai contextul furnizat."

prompt = "Ste úžasný asistent, ktorý hovorí po slovensky. Vašou úlohou je opraviť akékoľvek pravopisné rozdiely v prepísanom texte. Pridajte len potrebné interpunkčné znamienka, ako sú bodky, čiarky a veľké písmená a použite len poskytnutý kontext."



file_name= arg_source
audio_file = open(file_name, "rb")
transcript = client.audio.transcriptions.create(
  model=arguments.model, 
  language= arg_to,
  file=audio_file, 
  response_format="text"
)

with open(file_name + "_" + arg_to + ".txt", 'w') as ftxt:
    ftxt.write(transcript)
