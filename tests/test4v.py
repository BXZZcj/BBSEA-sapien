from openai import AzureOpenAI
api_key="c5ae1c3ed4e74e209fbb45cfc8cb3b2f"
azure_endpoint="https://gpt4v-0.openai.azure.com/"
client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-12-01-preview",
    azure_endpoint=azure_endpoint
)

response = client.chat.completions.create(
  model="vision",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image? 并使用中文回答"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])
