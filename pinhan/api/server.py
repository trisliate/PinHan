"""
PinHan FastAPI 服务

提供 RESTful API 接口
"""

import os
import time
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pinhan.engine import IMEEngineV3, create_engine_v3, EngineConfig, get_api_logger

# 初始化日志
logger = get_api_logger()


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
    
    logger.info("=" * 50)
    logger.info("PinHan API 服务启动")
    logger.info("正在初始化 IME 引擎...")
    
    # 词典目录：自动查找项目根目录下的 data/dicts
    config = EngineConfig(top_k=20)
    engine = create_engine_v3(config)  # 自动查找 data/dicts
    
    logger.info("IME 引擎初始化完成")
    logger.info(f"  词典服务: {'✓' if engine.dict_service else '✗'}")
    logger.info("=" * 50)
    
    yield
    
    logger.info("正在关闭 IME 引擎...")
    engine = None
    logger.info("PinHan API 服务已停止")


# ===== FastAPI 应用 =====
app = FastAPI(
    title="PinHan API",
    description="轻量级智能拼音输入法引擎 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 请求日志中间件 =====

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求的详细日志"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    
    # 请求信息
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    query = str(request.query_params) if request.query_params else ""
    
    logger.info(f"[{request_id}] --> {method} {path} {query} | IP: {client_ip}")
    
    try:
        response = await call_next(request)
        
        # 响应信息
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        status_code = response.status_code
        
        log_level = "info" if status_code < 400 else "warning" if status_code < 500 else "error"
        getattr(logger, log_level)(
            f"[{request_id}] <-- {status_code} | {elapsed_ms:.2f}ms"
        )
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        
        return response
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"[{request_id}] <-- ERROR | {elapsed_ms:.2f}ms | {type(e).__name__}: {e}")
        raise


# ===== API 路由 =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    from pinhan import __version__
    return HealthResponse(
        status="healthy" if engine else "not_ready",
        version=__version__,
    )


@app.post("/ime", response_model=IMEResponse)
async def process_pinyin(request: IMERequest):
    """处理拼音输入，返回汉字候选"""
    global engine
    
    if engine is None:
        logger.error("引擎未就绪，拒绝请求")
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    if not request.pinyin.strip():
        logger.warning(f"无效请求: 空拼音")
        raise HTTPException(status_code=400, detail="拼音不能为空")
    
    original_top_k = engine.config.top_k
    engine.config.top_k = request.top_k
    
    try:
        start = time.perf_counter()
        result = engine.process(request.pinyin, request.context)
        elapsed = (time.perf_counter() - start) * 1000
        
        # 详细日志
        top_candidates = [c.text for c in result.candidates[:3]]
        logger.debug(
            f"IME 处理: '{request.pinyin}' "
            f"| context='{request.context[:10]}...' "
            f"| top3={top_candidates} "
            f"| {elapsed:.2f}ms"
        )
        
        return IMEResponse(
            raw_pinyin=result.raw_pinyin,
            segmented_pinyin=result.segmented_pinyin,
            candidates=[
                CandidateItem(text=c.text, score=c.score, source=c.source)
                for c in result.candidates
            ],
            metadata=result.metadata,
        )
    except Exception as e:
        logger.error(f"IME 处理失败: pinyin='{request.pinyin}', error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        engine.config.top_k = original_top_k


@app.get("/ime/simple")
async def simple_query(pinyin: str, top_k: int = 10):
    """简单查询接口"""
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    result = engine.process(pinyin)
    candidates = [c.text for c in result.candidates[:top_k]]
    
    logger.debug(f"简单查询: '{pinyin}' -> {candidates[:3]}")
    
    return {"pinyin": pinyin, "candidates": candidates}


@app.get("/stats")
async def get_stats():
    """获取引擎统计信息"""
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎未就绪")
    
    stats = engine.get_stats()
    logger.info(f"统计查询: {stats}")
    
    return stats


# ===== 启动入口 =====

def main():
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"启动 PinHan API 服务: http://{host}:{port}")
    logger.info(f"API 文档: http://{host}:{port}/docs")
    logger.info(f"日志级别: {log_level.upper()}")
    
    uvicorn.run(
        "pinhan.api.server:app", 
        host=host, 
        port=port, 
        reload=False, 
        workers=1,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
