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

from engine import IMEEngine, create_engine
from config import EngineConfig


# ===== 请求/响应模型 =====

class IMERequest(BaseModel):
    """输入法请求"""
    pinyin: str = Field(..., description="拼音输入", example="nihaoma")
    context: str = Field("", description="上下文（已确认的文本）", example="你好")
    top_k: int = Field(10, description="返回候选数量", ge=1, le=50)


class CandidateItem(BaseModel):
    """候选项"""
    text: str = Field(..., description="汉字文本")
    score: float = Field(..., description="得分")
    source: str = Field(..., description="来源")


class IMEResponse(BaseModel):
    """输入法响应"""
    raw_pinyin: str = Field(..., description="原始输入")
    corrected_pinyin: str = Field(..., description="纠正后的拼音")
    segmented_pinyin: List[str] = Field(..., description="切分后的拼音")
    candidates: List[CandidateItem] = Field(..., description="候选列表")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    modules: dict


# ===== 全局引擎实例 =====
engine: Optional[IMEEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine
    
    # 启动时初始化引擎
    print("正在初始化 IME 引擎...")
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    config = EngineConfig(
        top_k=20,
        enable_corrector=True,
        enable_slm_rerank=True,
    )
    
    engine = create_engine(config, model_dir=project_root)
    print("IME 引擎初始化完成")
    
    yield
    
    # 关闭时清理
    print("正在关闭 IME 引擎...")
    engine = None


# ===== FastAPI 应用 =====
app = FastAPI(
    title="IME-SLM API",
    description="拼音输入法引擎 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== API 路由 =====

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    global engine
    
    modules = {
        "dict_service": engine.dict_service is not None if engine else False,
        "corrector": engine.corrector is not None if engine else False,
        "segmenter": engine.segmenter is not None if engine else False,
        "p2h_model": engine.p2h_model is not None if engine else False,
        "slm_model": engine.slm_model is not None if engine else False,
    }
    
    return HealthResponse(
        status="healthy" if engine else "not_ready",
        version="0.1.0",
        modules=modules,
    )


@app.post("/ime", response_model=IMEResponse, tags=["输入法"])
async def process_pinyin(request: IMERequest):
    """
    处理拼音输入，返回汉字候选
    
    - **pinyin**: 拼音输入（可以是连续的如 "nihaoma"，也可以空格分隔如 "ni hao ma"）
    - **context**: 上下文文本，用于语义重排
    - **top_k**: 返回候选数量
    """
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    if not request.pinyin.strip():
        raise HTTPException(status_code=400, detail="拼音不能为空")
    
    # 临时调整 top_k
    original_top_k = engine.config.top_k
    engine.config.top_k = request.top_k
    
    try:
        result = engine.process(request.pinyin, request.context)
        
        return IMEResponse(
            raw_pinyin=result.raw_pinyin,
            corrected_pinyin=result.corrected_pinyin,
            segmented_pinyin=result.segmented_pinyin,
            candidates=[
                CandidateItem(text=c.text, score=c.score, source=c.source)
                for c in result.candidates
            ],
        )
    finally:
        engine.config.top_k = original_top_k


@app.get("/ime/simple", tags=["输入法"])
async def simple_query(pinyin: str, top_k: int = 10):
    """
    简单查询接口（GET 方式）
    
    示例: /ime/simple?pinyin=nihao&top_k=5
    """
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    if not pinyin.strip():
        raise HTTPException(status_code=400, detail="拼音不能为空")
    
    original_top_k = engine.config.top_k
    engine.config.top_k = top_k
    
    try:
        result = engine.process(pinyin)
        
        return {
            "pinyin": pinyin,
            "candidates": [c.text for c in result.candidates],
        }
    finally:
        engine.config.top_k = original_top_k


@app.get("/dict/char/{pinyin}", tags=["词典"])
async def query_char(pinyin: str, limit: int = 10):
    """查询单字"""
    global engine
    
    if engine is None or engine.dict_service is None:
        raise HTTPException(status_code=503, detail="词典服务未就绪")
    
    chars = engine.dict_service.get_chars(pinyin)
    return {
        "pinyin": pinyin,
        "chars": [
            {"char": c, "freq": engine.dict_service.get_char_freq(c)} 
            for c in chars[:limit]
        ],
    }


@app.get("/dict/word/{pinyin}", tags=["词典"])
async def query_word(pinyin: str, limit: int = 10):
    """查询词语（拼音可用空格分隔，如 ni hao）"""
    global engine
    
    if engine is None or engine.dict_service is None:
        raise HTTPException(status_code=503, detail="词典服务未就绪")
    
    # 分割拼音
    pinyin_list = pinyin.split()
    if not pinyin_list:
        pinyin_list = [pinyin]
    
    words = engine.dict_service.get_words(pinyin_list)
    return {
        "pinyin": pinyin,
        "words": [
            {"word": w, "freq": engine.dict_service.get_word_freq(w)} 
            for w in words[:limit]
        ],
    }


# ===== 启动入口 =====

def main():
    """启动服务"""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"启动 IME-SLM API 服务: http://{host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
