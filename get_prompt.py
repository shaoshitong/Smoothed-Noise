import openai

openai.api_key = "8f565b39d2f34f728677e049984707a98296cee4fa754ab4ba3b14f627533aad"
openai.api_base = 'https://gpt-api.hkust-gz.edu.cn/v1/chat/completions'
original_prompts = []

with open("ucf_prompt.txt","r+") as f:
    m = f.readlines()
    original_prompts = [i.strip() for i in m]
    

result_prompts = []

for original_prompt in original_prompts:
    prompt = f"Please be a little more specific with the description of the video below, and please give feedback on the results only (i.e. don't reply with any redundant words such as sure, Great preparation and recovery, etc.) and give 10 feedbacks: {original_prompt}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4", # gpt-4, gpt-4-0613, gpt-4-1106-preview, gpt-4-32k, gpt-4-32k-0613
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )

    k = response.choices[0].message['content']
    k = k.replace("1. ","").replace("2. ","").replace("3. ","").replace("4. ","").replace("5. ","").replace("6. ","").replace("7. ","").replace("8. ","").replace("9. ","").replace("10. ","").split("\n")
    print(k)
    result_prompts.append(k)

with open("after_ucf_prompt.txt","w") as f:
    for result_prompt in result_prompts[:2]:
        for k in result_prompt:
            f.write(k+"\n")
        f.write("\n")