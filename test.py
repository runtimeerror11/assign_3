from google import genai
client = genai.Client(api_key="AIzaSyAETxBBtB5NP0Igsw_5qgra_I2HtHOhfxg")
response = client.models.generate_content(
   model="gemini-2.5-flash-lite",
    contents="Say hello in one sentence."
)
print(response.text)