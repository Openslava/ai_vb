from openai import OpenAI

client = OpenAI()

# tts-1-hd vs  tts-1
# alloy, echo, fable, onyx, nova, and shimmer
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="onyx",
    input='''Akákoľvek udalosť sa stáva konfliktnou práve tým, že jej prisúdime určitý negatívny význam v našej vedomej mysli
(tá istá udalosť alebo okolnosť môže spustiť u rôznych ľudí rôznu reakciu a tým aj odlišný biologický program)
Od okamihu vyriešenia konfliktu sa mobilizuje celý organizmus, aby obnovil pôvodnú funkciu postihnutého orgánu
Autonómny nervový systém sa prepne do parasympatickej činnosti, ktorá núti organizmus odpočívať, zatiaľ čo nás príroda lieči
'''
)

response.stream_to_file("output1.mp3")
response.stream_to_file