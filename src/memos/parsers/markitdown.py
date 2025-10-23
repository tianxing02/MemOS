import asyncio
import shlex
import time

from pathlib import Path

import httpx

from memos.configs.parser import MarkItDownParserConfig
from memos.log import get_logger
from memos.parsers.base import BaseParser


logger = get_logger(__name__)

BASE_URL = "http://106.75.235.231:8001"
LOCAL_TOKEN = "local_only_a8f3d2c1b5e7f9a6c4d8e2b1f7c3a9e5d2f8b4c6a1e9d7f3c5b8a2e4f6d9c1a3"
HEADERS = {"Authorization": f"Bearer {LOCAL_TOKEN}"}


async def check_service_health(client: httpx.AsyncClient):
    try:
        response = await client.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        print("✅ API服务运行正常")
        return True
    except (httpx.ConnectError, httpx.HTTPStatusError) as e:
        print(f"❌ API服务异常: {e}")
        return False


async def upload_ppt_file(client: httpx.AsyncClient, file_path: Path) -> str:
    print(f"\n📤 上传PPT文件: {file_path}")
    print(f"   文件大小: {file_path.stat().st_size / 1024 / 1024:.1f} MB")

    external_file_id = f"ppt-test-{int(time.time())}"

    print("🔄 使用直接文件上传接口...")

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/vnd.ms-powerpoint")}
        data = {
            "fileId": external_file_id,
            "force_ocr": "false",
            "ocr_all_images": "true",
        }

        print("📡 发送上传请求到 /api/file/upload...")
        response = await client.post(
            f"{BASE_URL}/api/file/upload", files=files, data=data, headers=HEADERS
        )

    response.raise_for_status()

    result = response.json()
    print(f"📄 上传响应: {result}")

    if result["code"] != 200:
        raise Exception(f"API返回错误: {result.get('message', '未知错误')}")

    generated_ids = result["data"]["generated_ids"]
    if len(generated_ids) == 0:
        raise Exception(f"服务器未生成文件ID，响应: {result}")

    file_id = generated_ids[0]
    print(f"🆔 获得内部file_id: {file_id}")
    return file_id


async def poll_status(client: httpx.AsyncClient, file_id: str):
    print(f"\n⏳ 监控处理状态: {file_id}")
    print("   注意: PPT文件处理需要更长时间（LibreOffice转换 + AI处理）")

    max_retries = 720
    poll_interval = 5

    for i in range(max_retries):
        response = await client.get(f"{BASE_URL}/api/v6/status/{file_id}", headers=HEADERS)
        response.raise_for_status()

        status_data = response.json()
        status = status_data.get("status")
        print(f"  - 尝试 {i + 1}/{max_retries}: 状态 '{status}'")

        if status == "completed":
            print("✅ 处理完成！")
            return
        elif status == "failed":
            error_msg = status_data.get("error_message", "未知错误")
            raise Exception(f"处理失败: {error_msg}")

        await asyncio.sleep(poll_interval)

    raise Exception("状态轮询超时")


async def download_and_verify_ppt(client: httpx.AsyncClient, file_id: str, temp_dir: Path):
    """下载并验证PPT处理结果"""
    print(f"\n📥 下载处理结果: {file_id}")

    temp_dir.mkdir(exist_ok=True, parents=True)

    archive_path = temp_dir / f"{file_id}.tar.zstd"
    curl_cmd = [
        "curl",
        "-o",
        str(archive_path),
        "-H",
        f"Authorization: Bearer {LOCAL_TOKEN}",
        f"{BASE_URL}/api/v6/download/{file_id}",
    ]

    proc = await asyncio.create_subprocess_exec(*curl_cmd)
    await proc.wait()

    if proc.returncode != 0:
        raise Exception(f"curl下载失败，返回码: {proc.returncode}")

    file_size = archive_path.stat().st_size
    print(f"📦 压缩包已保存: {archive_path} ({file_size} 字节)")

    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir()

    print("🔄 解压缩文件 (macOS兼容方式)...")
    decompress_cmd = (
        f"zstd -d < {shlex.quote(str(archive_path))} | tar -xf - -C {shlex.quote(str(extract_dir))}"
    )

    proc = await asyncio.create_subprocess_shell(
        decompress_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        stderr_text = stderr.decode() if stderr else "未知错误"
        print(f"⚠️ 管道方式失败，尝试分步解压: {stderr_text}")

        temp_tar = temp_dir / f"{file_id}.tar"

        zstd_cmd = ["zstd", "-d", str(archive_path), "-o", str(temp_tar)]
        proc = await asyncio.create_subprocess_exec(*zstd_cmd)
        await proc.wait()

        if proc.returncode != 0:
            raise Exception(f"zstd解压失败，返回码: {proc.returncode}")

        tar_cmd = ["tar", "-xf", str(temp_tar), "-C", str(extract_dir)]
        proc = await asyncio.create_subprocess_exec(*tar_cmd)
        await proc.wait()

        if proc.returncode != 0:
            raise Exception(f"tar解压失败，返回码: {proc.returncode}")

        temp_tar.unlink(missing_ok=True)

    print(f"📂 文件已解压到: {extract_dir}")

    result_content_dir = extract_dir / file_id

    md_files = list(result_content_dir.glob("**/*.md"))
    if not md_files:
        raise Exception("未找到markdown结果文件")

    result_file = md_files[0]
    print(f"✔️ 找到结果文件: {result_file}")

    content = result_file.read_text(encoding="utf-8")
    print(f"📄 提取的内容长度: {len(content)} 字符")

    lines = content.split("\n")

    text_content = ""
    for line in lines:
        if line.strip():
            text_content += line

    print("👍 PPT文件处理验证完成！")
    return text_content


async def cleanup_server_file(client: httpx.AsyncClient, file_id: str):
    print(f"\n🧹 清理服务器文件: {file_id}")
    response = await client.delete(f"{BASE_URL}/api/v6/delete/{file_id}", headers=HEADERS)
    response.raise_for_status()
    print("✅ 服务器清理完成")


class MarkItDownParser(BaseParser):
    """MarkItDown Parser class."""

    def __init__(self, config: MarkItDownParserConfig):
        self.config = config

    async def parse(self, file_path: str) -> str:
        ppt_file = Path(file_path)

        if not ppt_file.exists():
            print(f"❌ PPT文件不存在: {ppt_file}")
            return

        temp_dir = Path(f"./ppt_test_result/ppt_test_results_{int(time.time())}")
        temp_dir.mkdir(exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 0. 健康检查
                if not await check_service_health(client):
                    print("中止测试")
                    return

                file_id = await upload_ppt_file(client, ppt_file)

                await poll_status(client, file_id)

                text_content = await download_and_verify_ppt(client, file_id, temp_dir)

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            print(f"🔍 文件保留在代码目录用于检查: ./{temp_dir.name}/")
            return

        try:
            if file_id:
                async with httpx.AsyncClient() as client:
                    await cleanup_server_file(client, file_id)

            print(f"📂 下载的文件保留在代码目录: ./{temp_dir.name}/")
            print(f"   - 压缩包: ./{temp_dir.name}/file_*.tar.zstd")
            print(f"   - 解压内容: ./{temp_dir.name}/extracted/")
            print("💡 您可以在代码目录下直接检查处理结果和文件质量")

        except Exception as e:
            print(f"⚠️ 服务器清理失败: {e}")

        return text_content
