import requests
import sys
import time

# API 地址
API_URL = "http://localhost:6006/chat"

def main():
    print("="*50)
    print("本地部署 Qwen 终端对话客户端")
    print("输入 'exit' 或 'quit' 退出对话")
    print("输入 'clear' 清空对话历史")
    print("="*50)

    # 本地维护对话历史
    history = []

    while True:
        try:
            # 获取用户输入
            query = input("\nUser: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit']:
                print("再见！")
                break
                
            if query.lower() == 'clear':
                history = []
                print("--- 对话历史已清空 ---")
                continue

            # 构造请求数据
            payload = {
                "query": query,
                "history": history
            }

            print("Assistant: (思考中...)", end="\r")
            
            # 记录开始时间
            start_time = time.time()
            
            # 发送请求给 API Server
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "")
                    
                    # 清除"思考中"的提示
                    sys.stdout.write("\033[K") 
                    print(f"Assistant: {answer}")
                    
                    # 打印耗时
                    cost_time = time.time() - start_time
                    print(f"\033[90m(耗时: {cost_time:.2f}s)\033[0m")
                    
                    # 更新历史记录
                    history.append({"role": "user", "content": query})
                    history.append({"role": "assistant", "content": answer})
                    
                else:
                    print(f"\nError: API 返回错误 {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                print("\nError: 无法连接到 API 服务器。请确认 api_server.py 是否已启动。")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break

if __name__ == "__main__":
    main()
