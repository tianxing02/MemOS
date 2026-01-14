import json
import os
import re
import sys
import time
import uuid

from contextlib import suppress
from datetime import datetime

import requests

from dotenv import load_dotenv
from typing import Optional


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


class ZepClient:
    def __init__(self):
        from zep_cloud.client import Zep

        api_key = os.getenv("ZEP_API_KEY")
        self.client = Zep(api_key=api_key)

    def add(self, messages, user_id, timestamp):
        iso_date = datetime.fromtimestamp(timestamp).isoformat()
        for msg in messages:
            self.client.graph.add(
                data=msg.get("role") + ": " + msg.get("content"),
                type="message",
                created_at=iso_date,
                group_id=user_id,
            )

    def search(self, query, user_id, top_k):
        search_results = (
            self.client.graph.search(
                query=query, group_id=user_id, scope="nodes", reranker="rrf", limit=top_k
            ),
            self.client.graph.search(
                query=query, group_id=user_id, scope="edges", reranker="cross_encoder", limit=top_k
            ),
        )

        nodes = search_results[0].nodes
        edges = search_results[1].edges
        return nodes, edges


class Mem0Client:
    def __init__(self, enable_graph=False):
        from mem0 import MemoryClient

        self.client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        self.enable_graph = enable_graph

    def add(self, messages, user_id, timestamp, batch_size=2):
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            if self.enable_graph:
                self.client.add(
                    messages=batch_messages,
                    timestamp=timestamp,
                    user_id=user_id,
                    enable_graph=True,
                )
            else:
                self.client.add(
                    messages=batch_messages,
                    timestamp=timestamp,
                    user_id=user_id,
                    infer=False,
                )

    def search(self, query, user_id, top_k):
        res = self.client.search(
            query=query,
            top_k=top_k,
            user_id=user_id,
            enable_graph=self.enable_graph,
            filters={"AND": [{"user_id": f"{user_id}"}]},
        )
        return res


class MemobaseClient:
    def __init__(self):
        from memobase import MemoBaseClient

        self.client = MemoBaseClient(
            project_url=os.getenv("MEMOBASE_PROJECT_URL"), api_key=os.getenv("MEMOBASE_API_KEY")
        )

    def add(self, messages, user_id, batch_size=2):
        """
        messages = [{"role": "assistant", "content": data, "created_at": iso_date}]
        """
        from memobase import ChatBlob

        real_uid = self.string_to_uuid(user_id)
        user = self.client.get_or_create_user(real_uid)
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    _ = user.insert(ChatBlob(messages=batch_messages), sync=True)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

    def search(self, query, user_id, top_k):
        real_uid = self.string_to_uuid(user_id)
        user = self.client.get_user(real_uid, no_get=True)
        memories = user.context(
            max_token_size=top_k * 100,
            chats=[{"role": "user", "content": query}],
            event_similarity_threshold=0.2,
            fill_window_with_events=True,
        )
        return memories

    def delete_user(self, user_id):
        from memobase.error import ServerError

        real_uid = self.string_to_uuid(user_id)
        with suppress(ServerError):
            self.client.delete_user(real_uid)

    def string_to_uuid(self, s: str, salt="memobase_client"):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s + salt))


class MemosApiClient:
    """Product Add API 封装"""

    def __init__(self, timeout: float = 600.0):
        self.base_url = os.getenv("MEMOS_URL")
        self.headers = {"Content-Type": "application/json"}
        self.timeout = timeout

    def add(
        self,
        messages,
        user_id,
        writable_cube_ids: list[str],
        source_type: str,
        mode: str,
        async_mode: str,
    ):
        """
        调用 /product/add 接口

        Args:
            messages: 添加记忆信息
            user_id: 用户ID
            writable_cube_ids: 可写cube ID列表
            source_type: 来源类型
            mode: 模式 (fine/coarse)
            async_mode: 异步模式 (sync/async)
        """
        url = f"{self.base_url}/product/add"

        payload = {
            "user_id": user_id,
            "writable_cube_ids": writable_cube_ids,
            "messages": messages,
            "info": {"source_type": source_type},
            "mode": mode,
            "async_mode": async_mode,
        }

        response = requests.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self.headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        body = response.json()
        if body.get("code") is not None and body.get("code") != 200:
            raise RuntimeError(f"BUSINESS ERROR {body.get('code')}: {response.text}")

        return body

    def search(self, query, user_id, readable_cube_ids: list[str], top_k: str, mode: str):
        """
        调用 /product/search 接口

        Args:
            query: 搜索查询
            user_id: 用户ID
            readable_cube_ids: 可读cube ID列表, 默认为[user_id]
            top_k: 返回结果数量
        """

        url = f"{self.base_url}/product/search"

        if readable_cube_ids is None:
            readable_cube_ids = [user_id]

        payload = {
            "query": query,
            "user_id": user_id,
            "readable_cube_ids": readable_cube_ids,
            "top_k": top_k,
            "mode": mode,
        }

        response = requests.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self.headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        return response.json()


class MemosApiOnlineClient:
    def __init__(self):
        self.memos_url = os.getenv("MEMOS_ONLINE_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {os.environ['MEMOS_API_KEY']}",
        }

    def add(
        self,
        messages,
        user_id,
        writable_cube_ids: list[str],
        source_type: str,
        mode: str,
        async_mode: str,
    ):
        url = f"{self.memos_url}/add/message"
        payload = json.dumps(
            {
                "user_id": user_id,
                "conversation_id": user_id,
                "messages": messages,
                "writable_cube_ids": writable_cube_ids,
                "info": {"source_type": source_type},
                "mode": mode,
                "async_mode": async_mode,
            }
        )

        response = requests.request("POST", url, data=payload, headers=self.headers)
        assert response.status_code == 200, response.text
        assert json.loads(response.text)["message"] == "ok", response.text
        return response.json()

    def search(self, query: str, user_id: str, top_k: int, mode: str, knowledgebase_ids: list[str]):
        """Search memories."""
        url = f"{self.memos_url}/search/memory"
        data = {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": top_k,
            "knowledgebase_ids": knowledgebase_ids,
            "mode": mode,
        }

        resp = requests.post(url, headers=self.headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def upload_file(self, knowledgebase_id: str, file_url: str):
        """Upload file."""
        url = f"{self.memos_url}/add/knowledgebase-file"
        data = {
            "knowledgebase_id": knowledgebase_id,
            "file": [
                {
                    "content": file_url,
                }
            ],
        }

        resp = requests.post(url, headers=self.headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def check_file(self, file_ids: list[str]):
        """Check file state."""
        url = f"{self.memos_url}/get/knowledgebase-file"
        data = {"file_ids": file_ids}
        resp = requests.post(url, headers=self.headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()


class SupermemoryClient:
    def __init__(self):
        from supermemory import Supermemory

        self.client = Supermemory(api_key=os.getenv("SUPERMEMORY_API_KEY"))

        self.api_key = os.getenv("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SUPERMEMORY_API_KEY environment variable is not set. Please set it in your .env file or environment."
            )
        self.add_url = "https://api.supermemory.ai/v3/documents"
        self.search_url = "https://api.supermemory.ai/v3/search"

    def _sanitize_tag(self, s: str) -> str:
        t = str(s).strip()
        t = os.path.splitext(t)[0]
        t = t.replace(" ", "_")
        t = re.sub(r"[^A-Za-z0-9_-]", "_", t)
        t = re.sub(r"[_-]+", "_", t)
        t = t.strip("_")
        t = t.lower()
        if not re.match(r"^[a-z0-9]", t or ""):
            t = f"tag_{t}" if t else "tag_default"
        return t

    def add(self, content: Optional[str] = None, user_id: Optional[str] = None, messages: Optional[list] = None):
        if messages:
            content = "\n".join(
                f"{msg.get('chat_time', '')} {msg.get('role', '')}: {msg.get('content', '')}"
                for msg in messages
            )

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.client.memories.add(content=content, container_tag=user_id)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e
            return

        payload = {
            "content": content,
            "raw": content,
            "containerTag": self._sanitize_tag(user_id),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self.add_url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, user_id: str, top_k: int):
        payload = {
            "q": query,
            "limit": top_k,
            "containerTags": [self._sanitize_tag(user_id)],
            "rerank": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.search_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        chunk_list = []
        res = [entry.get("chunks") for entry in data.get("results", [])]
        for chunks in res:
            for chunk in chunks:
                chunk_list.append(chunk["content"])

        return chunk_list


class MemuClient:
    def __init__(self):
        from memu import MemuClient

        self.memu_client = MemuClient(
            base_url="https://api.memu.so", api_key=os.getenv("MEMU_API_KEY")
        )
        self.agent_id = "assistant_001"

    def add(self, messages, user_id, iso_date):
        try:
            response = self.memu_client.memorize_conversation(
                conversation=messages,
                user_id=user_id,
                user_name=user_id,
                agent_id=self.agent_id,
                agent_name=self.agent_id,
                session_date=iso_date,
            )
            self.wait_for_completion(response.item_id)
        except Exception as error:
            print("❌ Error saving conversation:", error)

    def search(self, query, user_id, top_k):
        user_memories = self.memu_client.retrieve_related_memory_items(
            user_id=user_id, agent_id=self.agent_id, query=query, top_k=top_k, min_similarity=0.1
        )
        res = [m.memory.content for m in user_memories.related_memories]
        return res

    def wait_for_completion(self, task_id):
        while True:
            status = self.memu_client.get_task_status(task_id)
            if status.status in ["SUCCESS", "FAILURE", "REVOKED"]:
                break
            time.sleep(2)


class FastGPTClient:
    def __init__(self):
        self.base_url = os.getenv("FASTGPT_BASE_URL")
        self.api_key = os.getenv("FASTGPT_API_KEY")

    def create_dataset(self, dataset_name: str):
        # 创建一个知识库
        url = f"{self.base_url}/core/dataset/create"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "name": dataset_name,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        datasetId = resp.json()["data"]
        return datasetId

    def delete_dataset(self, datasetId: str):
        # 删除特定ID的知识库（每个fastgpt账号创建的知识库数量受限制，因此需要删除测试完毕的知识库再创建新的）
        url = f"{self.base_url}/core/dataset/delete?id={datasetId}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.delete(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def add_content(self, datasetId: str, content: str, collection_name: str):
        url = f"{self.base_url}/core/dataset/collection/create/text"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "text": content,
            "datasetId": datasetId,
            "name": collection_name,
            "trainingType": "chunk",
            "chunkSettingMode": "auto",
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def upload_file(self, datasetId: str, file_url: str):
        url = f"{self.base_url}/proApi/core/dataset/collection/create/externalFileUrl"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "externalFileUrl": file_url,
            "externalFileId": file_url,
            "datasetId": datasetId,
            "trainingType": "chunk",
            "chunkSize": 512,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def batch_add_content(self, collectionId: str, data: list[str]):
        # 在集合内部批量添加数据（纯文本形式，其他形式的数据添加参考https://doc.fastgpt.cn/docs/introduction/development/openapi/dataset#%E9%9B%86%E5%90%88）
        url = f"{self.base_url}/core/dataset/data/pushData"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"collectionId": collectionId, "data": [{"q": d} for d in data]}
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search(self, datasetId: str, query: str, top_k: int):
        # 在知识库内部进行检索
        url = f"{self.base_url}/core/dataset/searchTest"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"datasetId": datasetId, "text": query, "searchMode": "embedding"}
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()

        result = resp.json()
        data_list = result["data"]["list"]
        return data_list

    def create_collection(self, datasetId: str, collection_name: str):
        # 在知识库内部创建集合
        url = f"{self.base_url}/core/dataset/collection/create"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"datasetId": datasetId, "name": collection_name, "type": "virtual"}
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        collectionId = resp.json()["data"]
        return collectionId


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "杭州西湖有什么好玩的"},
        {"role": "assistant", "content": "杭州西湖有好多松鼠，还有断桥"},
    ]
    user_id = "lme_exper_user_default_499"
    iso_date = "2023-05-01T00:00:00.000Z"
    timestamp = 1682899200
    query = "杭州西湖有什么"
    top_k = 5

    # MEMOS-API
    client = SupermemoryClient()
    for m in messages:
        m["created_at"] = iso_date
    # client.add(messages, user_id, user_id)
    memories = client.search(query, user_id, top_k)
    print(memories)
