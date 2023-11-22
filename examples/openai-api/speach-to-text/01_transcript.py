#
#
# inicialisation 
# python -m venv openai-env
# source openai-env/bin/activate
# pip install --upgrade openai
# export OPENAI_API_KEY="..."
#

# convert audio to text

from openai import OpenAI
client = OpenAI()

prompt = "Ești un asistent de ajutor care vorbește românește. Sarcina dumneavoastră este să corectați orice discrepanțe de ortografie din textul transcris. Adăugați doar semnele de punctuație necesare, cum ar fi puncte, virgule și majuscule și utilizați numai contextul furnizat."

file_name= './build/test.mp3'
audio_file = open(file_name, "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  language="ro",
  file=audio_file, 
  response_format="text"
)

with open(file_name + ".transcipt.txt", 'w') as ftxt:
    ftxt.write(transcript)
