"""
IME-SLM FastAPI 服务

提供 RESTful API 接口
"""

import os
import sys
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine import IMEEngineV3, create_engine_v3, EngineConfig


# ===== 请求/响应模型 =====

class IMERequest(BaseModel):
    """输入法请求"""
    pinyin: str = Field(..., description="拼音输入")
    context: str = Field("", description="上下文（已确认的文本）")
    top_k: int = Field(10, description="返回候选数量", ge=1, le=50)


class CandidateItem(BaseModel):
    """候选项"""
    text: str
    score: float
    source: str


class IMEResponse(BaseModel):
    """输入法响应"""
    raw_pinyin: str
    segmented_pinyin: List[str]
    candidates: List[CandidateItem]
    metadata: Optional[dict] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str


# ===== 全局引擎实例 =====
engine: Optional[IMEEngineV3] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine
    
    print("正在初始化 IME 引擎...")
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    config = EngineConfig(top_k=20)
    engine = create_engine_v3(config, model_dir=project_root)
    print("IME 引擎初始化完成")
    
    yield
    
    print("正在关闭 IME 引擎...")
    engine = None


# ===== FastAPI 应用 =====
app = FastAPI(
    title="IME-SLM API",
    description="拼音输入法引擎 API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== API 路由 =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if engine else "not_ready",
        version="0.2.0",
    )


@app.post("/ime", response_model=IMEResponse)
async def process_pinyin(request: IMERequest):
    """处理拼音输入，返回汉字候选"""
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    if not request.pinyin.strip():
        raise HTTPException(status_code=400, detail="拼音不能为空")
    
    original_top_k = engine.config.top_k
    engine.config.top_k = request.top_k
    
    try:
        result = engine.process(request.pinyin, request.context)
        
        return IMEResponse(
            raw_pinyin=result.raw_pinyin,
            segmented_pinyin=result.segmented_pinyin,
            candidates=[
                CandidateItem(text=c.text, score=c.score, source=c.source)
                for c in result.candidates
            ],
            metadata=result.metadata,
        )
    finally:
        engine.config.top_k = original_top_k


@app.get("/ime/simple")
async def simple_query(pinyin: str, top_k: int = 10):
    """简单查询接口"""
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    result = engine.process(pinyin)
    return {"pinyin": pinyin, "candidates": [c.text for c in result.candidates[:top_k]]}


@app.get("/stats")
async def get_stats():
    """获取引擎统计信息"""
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    return engine.get_stats()


# ===== 启动入口 =====

def main():
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"启动 IME-SLM API 服务: http://{host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")
    
    uvicorn.run("api.server:app", host=host, port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()
