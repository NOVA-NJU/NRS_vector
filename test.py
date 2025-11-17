from services import VectorService
from models import DocumentPayload, VectorSearchRequest

svc = VectorService()

# 插入文档
doc1 = DocumentPayload(document_id="1", text="南京大学仙林校区图书馆开馆时间是上午八点。", metadata={})
doc2 = DocumentPayload(document_id="2", text="学校食堂今天中午提供烤肉饭。", metadata={})
import asyncio
asyncio.run(svc.upsert_document(doc1))
asyncio.run(svc.upsert_document(doc2))

# 检索
search_req = VectorSearchRequest(query="图书馆几点开门", top_k=2)
result = asyncio.run(svc.search(search_req))
print(result)