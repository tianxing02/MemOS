import os

from openai import OpenAI


# ===== 配置 =====
os.environ["OPENAI_API_KEY"] = "sk-5hTdYsmX7ErNsLQKJ8jUTahHzyDjQ3H5mpgAeTfUfeRyIUYV"
os.environ["OPENAI_BASE_URL"] = "http://123.129.219.111:3000/v1"

# ===== 创建客户端 =====
client = OpenAI()


def test_chat():
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "你好，简单介绍一下你自己"}],
            temperature=0.2,
            max_tokens=100,
        )

        print("\n✅ 请求成功")
        print("=" * 40)
        print(resp.choices[0].message.content)

    except Exception as e:
        print("\n❌ 请求失败：")
        print(e)


if __name__ == "__main__":
    test_chat()
