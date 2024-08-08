from openai import OpenAI
import os 

os.environ["OPENAI_API_BASE"] = "https://gpt-api.hkust-gz.edu.cn/v1"
os.environ["OPENAI_API_KEY"] = "8f565b39d2f34f728677e049984707a98296cee4fa754ab4ba3b14f627533aad"


client = OpenAI(  api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_API_BASE")
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
)

print(completion.choices[0].message)
