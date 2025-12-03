"""
PinHan 命令行工具
"""

import argparse
import sys


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        prog="pinhan",
        description="PinHan - 轻量级智能拼音输入法引擎",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # server 命令
    server_parser = subparsers.add_parser("server", help="启动 API 服务")
    server_parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认: 0.0.0.0)")
    server_parser.add_argument("--port", type=int, default=3000, help="端口 (默认: 3000)")
    
    # query 命令
    query_parser = subparsers.add_parser("query", help="查询拼音")
    query_parser.add_argument("pinyin", help="拼音输入")
    query_parser.add_argument("-c", "--context", default="", help="上下文")
    query_parser.add_argument("-k", "--top-k", type=int, default=5, help="返回候选数量")
    
    # version 命令
    subparsers.add_parser("version", help="显示版本")
    
    args = parser.parse_args()
    
    if args.command == "server":
        from pinhan.api.server import main as server_main
        import os
        os.environ["HOST"] = args.host
        os.environ["PORT"] = str(args.port)
        server_main()
    
    elif args.command == "query":
        from pinhan import create_engine_v3
        engine = create_engine_v3()
        result = engine.process(args.pinyin, args.context)
        for i, c in enumerate(result.candidates[:args.top_k], 1):
            print(f"{i}. {c.text} ({c.score:.3f})")
    
    elif args.command == "version":
        from pinhan import __version__
        print(f"PinHan v{__version__}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
