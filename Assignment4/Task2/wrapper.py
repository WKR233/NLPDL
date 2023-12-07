def llm(messages, **kwargs) -> str:

    import openai
    openai.api_key="sess-TB863soQiBOMZecdxUiD8mMbzJWB8yaa7U6rjSxL"

    response=openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user","content":messages}
        ]
    )

    return response.choices[0].message.content