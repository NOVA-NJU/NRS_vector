"""测试 LangChain 分块功能整合"""
import asyncio
from services import vector_service
from models import DocumentPayload, VectorSearchRequest

# 测试文本（来自 test.py）
test_text = """
亲爱的小蓝鲸：

在AI重塑世界的今天，我们应该如何与技术共存？在大学四年里，我们是否可以真正掌握知识、理解世界，获得将理想化为实践的能力？我们能否让手中的科技不仅改变自己，还能让身边每一位同学受益？

如果你渴望摆脱"内卷"，走出"知识焦虑"，寻找改变的路径：在南京大学学生智能数据决策工作室——NOVA，我们正在寻找未来的「改变者」与「颠覆者」。

NOVA以"超新星"命名，成立于2017年，最初是一群技术爱好者自发组织的学术沙龙。走过八年，NOVA已经成长为一个面向全校本科生开放、以"问题导向、合作加速、自主驱动"为核心的学生社团。

我们关注AI技术本身，更关注技术如何被合理使用、实际落地，解决真实问题。我们相信，在这个知识快速更迭、AI飞速发展的时代，大学生的核心竞争力是问题意识、自主学习能力和动手实践能力。

NOVA的核心理念：从"填充"到"创新"
NOVA的目标不只是传授知识，而是通过项目实践和团队协作，帮助你从"被动学习者"转变为"主动创造者"。我们倡导从填充式学习转向内驱式学习，通过批判性思维训练与创新实践，快速提高自主学习能力，孵化创新思维能力。

从"供给侧"入手：
1. 创新与实践相结合：参与从零开始的真实项目，如新生宿舍分配系统、新生校园卡照片替换系统、新生导师分配算法等。
2. 模块化思维与知识图谱：将"知识"从单一碎片化的信息点重构为结构化的思维框架。
3. 团队共享与思维碰撞：通过小组讨论、工作坊和分享会，实现多维度的思维碰撞和创新孵化。

从"需求侧"出发：
1. 个性化学习路径：从新手到初级、从初级到高级，逐步掌握核心技能。
2. 合理的时间管理：每周不少于10小时的学习任务，采用碎片化学习方式。
3. 快速反馈与迭代：通过任务汇总、问题梳理、阶段性反思和成果汇报。

NOVA资源：
1. 语雀知识平台：社员分享、项目文档、学习笔记在线开放协作
2. 社团技术资源：虚拟机、GPU、VPN等软硬件工具
3. 丰富的社团活动：每周技术沙龙、分享会、团队建设
4. 跨校区联动：鼓楼、仙林、苏州三地同步参与

在NOVA，你会收获：
1. 项目实战经验
2. 导师与前辈的支持
3. 广阔的发展空间
4. 志同道合的伙伴
5. 真实价值感

NOVA正在寻找：
愿意主动学习、愿意合作共享、愿意用行动去探索AI的「改变者」与「颠覆者」。专业不限，基础不限，欢迎所有愿意改变、愿意突破的25级本科新同学。

加入NOVA，你将不仅仅是技术的学习者，更是科技变革的推动者。让我们一起，用数字与科技，点亮校园的未来！

南京大学智能数据决策工作室（NOVA）——做校园科技的引领者，做改变者的聚集地！
"""

async def main():
    print("=" * 60)
    print("测试 1: 文档分块存储")
    print("=" * 60)
    
    # 创建文档
    payload = DocumentPayload(
        document_id="nova_recruitment_2025",
        text=test_text,
        metadata={"source": "微信公众号", "category": "社团招新"}
    )
    
    # 存储文档（如果启用分块会自动切分）
    result = await vector_service.upsert_document(payload)
    print(f"✓ 文档存储成功: {result.document_id}")
    print(f"✓ 存储状态: {result.status}")
    
    # 检查生成的块
    doc_keys = list(vector_service._documents.keys())
    chunk_keys = [k for k in doc_keys if "chunk" in k]
    
    if chunk_keys:
        print(f"\n✓ 启用了分块，生成了 {len(chunk_keys)} 个文本块:")
        for i, key in enumerate(chunk_keys[:3]):  # 只显示前3个
            chunk = vector_service._documents[key]
            print(f"  块 {i+1} [{key}]:")
            print(f"    长度: {len(chunk.text)} 字符")
            print(f"    元数据: parent_doc={chunk.metadata.get('parent_doc')}, "
                  f"chunk_index={chunk.metadata.get('chunk_index')}, "
                  f"total_chunks={chunk.metadata.get('total_chunks')}")
            print(f"    内容预览: {chunk.text[:60]}...")
    else:
        print(f"\n✓ 未启用分块，存储为完整文档")
        print(f"  文档ID: {doc_keys[0]}")
        print(f"  文本长度: {len(vector_service._documents[doc_keys[0]].text)} 字符")
    
    print("\n" + "=" * 60)
    print("测试 2: 语义搜索")
    print("=" * 60)
    
    # 测试搜索
    search_request = VectorSearchRequest(
        query="NOVA寻找什么样的同学？",
        top_k=3
    )
    
    search_result = await vector_service.search(search_request)
    print(f"\n✓ 搜索查询: {search_result.query}")
    print(f"✓ 返回结果数: {len(search_result.results)}")
    
    for i, match in enumerate(search_result.results):
        print(f"\n  结果 {i+1}:")
        print(f"    文档ID: {match.document_id}")
        print(f"    相似度: {match.score:.4f}")
        if "parent_doc" in match.metadata:
            print(f"    父文档: {match.metadata['parent_doc']}")
            print(f"    块索引: {match.metadata['chunk_index']}/{match.metadata['total_chunks']}")
        doc = vector_service._documents[match.document_id]
        if i == 0:
            print(f"    内容详情: {doc.text[:500]}")
        else:
            print(f"    内容预览: {doc.text[:100]}...")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
