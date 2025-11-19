"""测试搜索接口是否返回文档文本内容。"""
import asyncio
from models import DocumentPayload, VectorSearchRequest
from services import vector_service


async def test_search_returns_text():
    """测试搜索功能是否返回文本内容。"""
    print("=" * 60)
    print("测试搜索接口返回文档文本内容")
    print("=" * 60)
    
    # 1. 添加测试文档
    print("\n1. 添加测试文档...")
    test_doc = DocumentPayload(
        document_id="test_doc_001",
        text="南京大学仙林校区图书馆上午8点开放，晚上10点关闭。",
        metadata={"source": "测试", "category": "校园服务"}
    )
    
    response = await vector_service.upsert_document(test_doc)
    print(f"   ✓ 文档已存储: {response.document_id}")
    
    # 2. 执行搜索
    print("\n2. 执行语义搜索...")
    search_request = VectorSearchRequest(
        query="图书馆几点开门",
        top_k=1
    )
    
    search_response = await vector_service.search(search_request)
    print(f"   查询: {search_response.query}")
    print(f"   返回结果数: {len(search_response.results)}")
    
    # 3. 检查返回结果
    print("\n3. 检查搜索结果...")
    if search_response.results:
        result = search_response.results[0]
        print(f"   ✓ 文档 ID: {result.document_id}")
        print(f"   ✓ 相似度分数: {result.score}")
        print(f"   ✓ 文档文本: {result.text}")
        print(f"   ✓ 元数据: {result.metadata}")
        
        # 验证 text 字段存在且不为空
        if result.text:
            print("\n" + "=" * 60)
            print("✅ 测试通过！搜索接口成功返回文档文本内容")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ 测试失败：text 字段为空")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败：未返回任何搜索结果")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_search_returns_text())
