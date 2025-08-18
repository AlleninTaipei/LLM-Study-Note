# MCP - Model Context Protocol

* *MCP, you can consider it to be a layer between your llm and the services and the tools and this layer translates all those different languages into a unified language that makes complete sense to the llm. - [Source: Model Context Protocol (MCP), clearly explained (why it matters)](https://youtu.be/7j_NE6Pjv-E?si=Axh186THqo6FYUZT)*

* *[Greg Isenberg](https://www.youtube.com/@GregIsenberg) - This is the startup ideas YouTube channel. Hosted by Greg Isenberg (CEO Late Checkout, ex-advisor of Reddit, TikTok etc).*

* *[Ras Mic](https://www.youtube.com/@rasmic) - Full Stack Engineer & YouTuber.*

## General API Concept

|General API Concept|left Side|
|-|-|
|Client|Represents a user or application (e.g., a web browser) that wants to access data or functionality.|
|Backend (Server)|Represents the server-side application that provides the data or functionality.|
|API (Application Programming Interface)|A set of rules and protocols that allow different software applications to communi1cate with each other. In this diagram, the API acts as the intermediary between the Client and the Backend, and between the Backend and various Services.|
|Service #1 and Service #2|Represent external services that the Backend might need to interact with to fulfill the Client's request. These could be databases, other applications, or specialized functionalities.|

|REST API Concept|right side|
|-|-|
|Client|Similar to the left side, it's the user or application making the request.|
|HTTP (Hypertext Transfer Protocol)|The protocol used for communication over the internet. REST APIs typically use HTTP methods like GET, POST, PUT, DELETE to perform operations.|
|URL (Uniform Resource Locator)|The address of the resource being requested or manipulated.|
|Server|The server that hosts the data and handles the requests.|
|JSON (JavaScript Object Notation)|A common data format used for exchanging data between the Client and Server. It's human-readable and easily parsed by machines.|

In essence, the left side shows a general architecture where a Client interacts with a Backend (Server) through an API, and the Backend might use other APIs to interact with different Services. The right side explains what a REST API is – a specific type of API that uses HTTP, URLs, and JSON for communication over the internet.

![image](assets/generalapiconcept.png)

|API（應用程式介面）|左側|
|-|-|
|Client（客戶端）|代表想要存取資料或功能的用戶或應用程式（例如網頁瀏覽器）。|
|Backend (Server)（後端/伺服器）|代表提供資料或功能的伺服器端應用程式。|
|API（應用程式介面）|一組規則和協定，允許不同的軟體應用程式彼此通訊。 在此圖中，API 充當客戶端和後端之間，以及後端和各種服務之間的媒介。|
|Service #1 和 Service #2（服務 #1 和服務 #2）|代表後端可能需要互動才能滿足客戶端請求的外部服務。 這些可能是資料庫、其他應用程式或特殊功能。|

|REST API（具象狀態傳輸應用程式介面）|右側|
|-|-|
|Client（客戶端）|與左側類似，它是發出請求的用戶或應用程式。|
|HTTP（超文本傳輸協定）|用於在網際網路上進行通訊的協定。 REST API 通常使用 HTTP 方法，例如 GET、POST、PUT、DELETE 來執行操作。|
|URL（統一資源定位符）|被請求或操作的資源的地址。|
|Server（伺服器）|託管資料並處理請求的伺服器。|
|JSON（JavaScript 物件表示法）|用於在客戶端和伺服器之間交換資料的常用資料格式。 它是人類可讀的，並且可以被機器輕鬆解析。|

本質上，左側顯示了一個通用架構，其中客戶端透過 API 與後端（伺服器）互動，而後端可能使用其他 API 與不同的服務互動。 右側解釋了什麼是 REST API – 一種特定類型的 API，它使用 HTTP、URL 和 JSON 在網際網路上進行通訊。

---

## the evolution of Large Language Models (LLMs)

|Stages||
|-|-|
|Just the LLM by itself|This represents the basic LLM in isolation. The text above states that "LLMs by themselves are incapable of doing anything meaningful." This highlights the limitation of a raw LLM without any external tools or systems to interact with the real world.  It can process and generate text based on its training data, but it can't perform actions or access real-time information.|
|LLMs + Tools|This stage shows the LLM connected to external "Tools." These tools could be APIs, databases, or other software that allow the LLM to perform specific tasks. For example, a tool could be a search engine API to fetch real-time information, a calculator for mathematical operations, or a database to retrieve specific data.  This integration significantly expands the LLM's capabilities, allowing it to move beyond just text generation and into more practical applications.|
|LLMs + MCP|This stage introduces the concept of an "MCP" (likely standing for something like "Model Control Plane" or "Multi-Component Platform"). The MCP acts as an intermediary between the LLM and various "Services."  This suggests a more sophisticated architecture where the MCP manages the interaction between the LLM and multiple specialized services.  These services could be anything from specific data processing pipelines to user interface components. The MCP likely handles tasks like routing requests, managing data flow, and ensuring security and reliability.|

In essence, the image argues that LLMs become truly powerful when they are integrated with external tools and systems, moving from isolated text generators to components in a larger, more functional architecture.

![image](assets/stagesofllm.png)

|大型語言模型 (LLMs) 及其能力的演進||
|-|-|
|僅有 LLM 本身|這代表獨立運作的基本 LLM。 上方的文字說明「LLMs 本身無法做任何有意義的事情」。 這突顯了原始 LLM 的局限性，它沒有任何外部工具或系統與現實世界互動。 它可以根據其訓練數據處理和生成文本，但無法執行操作或訪問即時資訊。|
|LLMs + 工具|這個階段顯示 LLM 連接到外部「工具」。 這些工具可以是 API、資料庫或其他軟體，讓 LLM 執行特定任務。 例如，工具可以是搜尋引擎 API 來獲取即時資訊、用於數學運算的計算器或用於檢索特定資料的資料庫。 這種整合顯著擴展了 LLM 的能力，使其能夠超越僅僅生成文本，進入更實際的應用。|
|LLMs + MCP|這個階段引入了「MCP」（可能代表「模型控制平面」或「多組件平台」之類的東西）的概念。 MCP 充當 LLM 和各種「服務」之間的媒介。 這表明了一種更複雜的架構，其中 MCP 管理 LLM 和多個專用服務之間的互動。 這些服務可以是從特定的資料處理管道到使用者介面組件的任何東西。 MCP 可能處理諸如路由請求、管理資料流以及確保安全性和可靠性等任務。|

本質上，該圖片認為 LLM 與外部工具和系統整合後才會變得真正強大，從獨立的文本生成器轉變為更大、更具功能性架構中的組件。

---

## MCP Ecosystem Overview

|MCP Ecosystem Overview||
|-|-|
|MCP Client|Represents an application or component that wants to utilize a specific service. It initiates communication with the MCP Server. ["Tempo"](https://www.tempo.new/) and ["Windsurf"](https://codeium.com/) are given as examples of MCP Clients.|
|MCP Protocol|The communication protocol used between the MCP Client and the MCP Server. It defines the rules and format for exchanging messages.|
|MCP Server|Acts as an intermediary between the MCP Client and the Service. It likely manages requests, handles authentication, and ensures proper communication.|
|Service|The actual functionality or data that the MCP Client wants to access. It could be a database, an API, or any other external resource. The note "Dev tool company / Database" suggests the Service is provided by a development tool company or is a database.|
|"Maintained by Service Provider"|This note at the bottom indicates that the Service (and possibly the MCP Server) is managed and maintained by an external entity, the Service Provider. This implies that the MCP Client doesn't directly interact with the Service but goes through the MCP Server.|

In essence, the diagram shows a client-server architecture where the MCP Server acts as a gateway to a Service, using a defined protocol to facilitate communication. The service is maintained by a separate provider, indicating a separation of concerns and a potentially distributed system.

![image](assets/mcpworkflow.png)

|MCP 生態系統概述||
|-|-|
|MCP 客戶端 (MCP Client)|代表想要使用特定服務的應用程式或元件。 它發起與 MCP 伺服器的通訊。 ["Tempo"](https://www.tempo.new/) 和 ["Windsurf"](https://codeium.com/) 被作為 MCP 客戶端的範例。|
|MCP 協定 (MCP Protocol)|MCP 客戶端和 MCP 伺服器之間使用的通訊協定。 它定義了交換訊息的規則和格式。|
|MCP 伺服器 (MCP Server)|充當 MCP 客戶端和服務之間的媒介。 它可能管理請求、處理身份驗證並確保正確的通訊。|
|服務 (Service)|MCP 客戶端想要存取的實際功能或資料。 它可以是資料庫、API 或任何其他外部資源。 「Dev tool company / Database」（開發工具公司/資料庫）的註解表明該服務由開發工具公司提供或是一個資料庫。|
|「由服務提供者維護 (Maintained by Service Provider)」|底部的註解表示服務（以及可能的 MCP 伺服器）由外部實體（服務提供者）管理和維護。 這意味著 MCP 客戶端不會直接與服務互動，而是透過 MCP 伺服器進行互動。|

本質上，該圖顯示了一個客戶端-伺服器架構，其中 MCP 伺服器充當服務的閘道，使用定義的協定來促進通訊。 該服務由單獨的提供者維護，表示關注點的分離和潛在的分散式系統。

---

## Conclusion on MCP's Potential

* The host inquired about potential startup opportunities arising from [MCP](https://www.anthropic.com/news/model-context-protocol), similar to those seen with protocols like HTTPS and SMTP, and whether it matters to individuals developing ideas.

* Professor Ross Mike suggested: For technical individuals, there are many opportunities. He proposed the idea of an [MCP App Store](https://www.mcpappstore.com/), where developers could easily find, install, and deploy MCP servers from various repositories. For non-technical individuals, he advised staying updated on platforms building MCP capabilities and observing the evolution of the standards. He noted that integrating tools is currently difficult, but finalized MCP standards will lead to much smoother integration. However, he cautioned that it's still very early stages for MCP, and significant business decisions might be premature, as a different standard could emerge, potentially from a major player like OpenAI. He recommended observing and learning for now, and being ready to act when the right standard is finalized

![image](assets/samaltmanxmcp.png)
> OpenAI CEO Sam Altman said that OpenAI will add support for Anthropic’s Model Context Protocol, or MCP, across its products, including the desktop app for ChatGPT

* 主持人 Greg 最後問了一個問題，關於在 [MCP](https://www.anthropic.com/news/model-context-protocol) 這個協議普及之後，是否會像過去 HTTPS 或 SMTP 這樣的協議一樣，出現許多基於此協議的新創事業機會。他特別想知道這對於正在發展想法的聽眾是否有影響.

* 來賓 Professor Ross Mike 回答說，對於技術人員來說，他認為有很多機會。他舉了一個免費的想法，那就是可以創建一個 [MCP App Store](https://www.mcpappstore.com/)。他觀察到現在有很多 MCP 伺服器的程式碼儲存庫在各處，如果有人可以創建一個網站，讓使用者能夠瀏覽這些 MCP 伺服器，看到 GitHub 程式碼，然後可以點擊「安裝」或「部署」，這樣伺服器就會部署並給他們一個特定的 URL，他們可以將這個 URL 貼到 MCP 客戶端並開始使用. 他半開玩笑地說，如果有人因為這個想法賺了數百萬，只需要給他一千美元就好. 對於非技術人員來說，他建議關注那些正在建立 MCP 功能的平台，並留意最終的標準會是什麼。他提到雖然現在每週都有新的帶有工具的聊天機器人介面出現，但整合這些工具並不容易。他認為，一旦 MCP 的標準確定下來，並且服務提供商開始建構他們的 mCP 或類似的東西，非技術人員就能夠更無縫、更輕鬆地進行整合. 然而，他也指出，目前來看，無論是對於非技術人員還是技術人員，現在就採取重大的商業決策可能還為時過早，因為 MCP 仍處於非常早期的階段。他提到，如果像 OpenAI 這樣的公司明天提出一個新的標準，那麼現在的一切都可能會改變。因此，他建議大家現在要做的就是觀察、學習，等待時機成熟再採取行動。他認為，理解 MCP 的運作方式將有助於理解未來可能出現的新事物，並在最終標準確定時能夠迅速行動。

# MCP vs Function Call 的區別

## MCP (Model Context Protocol)

* Anthropic 開發的標準化協議，用於 AI 模型與外部系統之間的通信
* 提供了一個統一的接口，讓 LLM 能夠安全地訪問各種外部資源（文件系統、數據庫、API 等）
* 主要解決的是連接性問題 - 如何讓 AI 模型與外界系統溝通
* 包含了身份驗證、權限管理、資源發現等完整的協議棧

## Function Call
* **是 LLM 的一個核心能力**，指模型能夠識別何時需要調用外部函數並正確格式化調用
* 專注於函數調用的邏輯 - 理解何時調用、如何構造參數、如何處理返回值
* 是實現工具使用的基礎技術

## Function Call = Tool Call 嗎？

基本上是的，這兩個術語在實際使用中幾乎可以互換：
* Function Call 更技術性，強調調用函數的機制
* Tool Call 更概念性，強調使用工具來完成任務
* 在 OpenAI 的 API 中稱為 "function calling"
* 在 Anthropic 的 Claude 中稱為 "tool use"
* 本質上都是讓 LLM 調用預定義的外部功能

## 請用 "路徑" 來解釋一下

### 傳統 LLM 路徑

* **用戶問題 → LLM → 文字回答**

### 有 Function Call 的路徑

* **用戶問題 → LLM → 識別需要工具 → 調用函數 → 獲得結果 → LLM → 整合回答**

### MCP 框架下的完整路徑

用戶問題 
    ↓
   LLM (判斷需要外部資源)
    ↓
MCP 協議層 (身份驗證、權限檢查)
    ↓
外部系統連接 (數據庫/API/文件系統)
    ↓
Function Call 執行
    ↓
結果返回 → MCP 協議層
    ↓
   LLM (整合結果)
    ↓
  最終回答

## 核心差異

### Function Call：

* 直接的函數調用機制
* LLM 直接知道有哪些函數可用
* 簡單直接的執行路徑

### MCP：

* 有完整的連接、驗證、權限管理流程
* 支援多個伺服器和資源的統一管理
* 更安全、更標準化的架構


```python
# ============================================
# 1. 純 Function Call 示範 (OpenAI 風格)
# ============================================

import json
from openai import OpenAI

# 定義可用的函數
def get_weather(location: str) -> dict:
    """獲取天氣資訊"""
    # 模擬天氣 API 調用
    return {"location": location, "temperature": "25°C", "condition": "晴天"}

def calculate_math(expression: str) -> float:
    """計算數學表達式"""
    return eval(expression)  # 生產環境請勿使用 eval

# Function Call 的函數定義（給 LLM 看的）
available_functions = {
    "get_weather": {
        "name": "get_weather",
        "description": "獲取指定地點的天氣資訊",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名稱"}
            },
            "required": ["location"]
        }
    },
    "calculate_math": {
        "name": "calculate_math", 
        "description": "計算數學表達式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "數學表達式"}
            },
            "required": ["expression"]
        }
    }
}

# 實際的函數映射
function_map = {
    "get_weather": get_weather,
    "calculate_math": calculate_math
}

def handle_function_call(function_name: str, arguments: dict):
    """處理 LLM 的函數調用請求"""
    if function_name in function_map:
        return function_map[function_name](**arguments)
    else:
        raise ValueError(f"未知函數: {function_name}")

# 模擬 LLM 的 Function Call 流程
def simulate_llm_with_function_call(user_message: str):
    """模擬帶有 Function Call 的 LLM 對話"""
    
    # 第一步：LLM 判斷是否需要調用函數
    if "天氣" in user_message:
        # LLM 決定調用天氣函數
        function_call = {
            "name": "get_weather",
            "arguments": {"location": "台北"}
        }
        
        # 執行函數調用
        result = handle_function_call(function_call["name"], function_call["arguments"])
        
        # LLM 整合結果回答
        return f"根據查詢結果，{result['location']}現在是{result['condition']}，溫度{result['temperature']}"
    
    return "我可以幫你查天氣或計算數學問題！"
```

```python
# ============================================
# 2. MCP 框架示範
# ============================================

class MCPServer:
    """MCP 伺服器 - 提供資源和工具"""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.resources = {}
        self.tools = {}
        self.permissions = {}
    
    def register_resource(self, resource_id: str, resource_data):
        """註冊資源"""
        self.resources[resource_id] = resource_data
    
    def register_tool(self, tool_name: str, tool_function, description: str):
        """註冊工具"""
        self.tools[tool_name] = {
            "function": tool_function,
            "description": description
        }
    
    def authenticate(self, client_id: str) -> bool:
        """身份驗證"""
        # 簡化的驗證邏輯
        return client_id in ["trusted_llm", "claude", "gpt"]
    
    def check_permission(self, client_id: str, resource: str) -> bool:
        """權限檢查"""
        return self.authenticate(client_id)

class MCPClient:
    """MCP 客戶端 - LLM 端"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.connected_servers = {}
    
    def connect_to_server(self, server: MCPServer):
        """連接到 MCP 伺服器"""
        if server.authenticate(self.client_id):
            self.connected_servers[server.server_name] = server
            return f"成功連接到 {server.server_name}"
        else:
            raise PermissionError("身份驗證失敗")
    
    def discover_resources(self, server_name: str):
        """發現可用資源"""
        if server_name in self.connected_servers:
            server = self.connected_servers[server_name]
            return list(server.resources.keys())
        return []
    
    def call_tool(self, server_name: str, tool_name: str, **kwargs):
        """通過 MCP 調用工具"""
        server = self.connected_servers[server_name]
        
        # MCP 協議處理
        if not server.check_permission(self.client_id, tool_name):
            raise PermissionError("無權限訪問此工具")
        
        if tool_name not in server.tools:
            raise ValueError(f"工具 {tool_name} 不存在")
        
        # 執行實際的函數調用
        tool_function = server.tools[tool_name]["function"]
        return tool_function(**kwargs)

# ============================================
# 3. 實際使用示範
# ============================================

def demo_function_call_only():
    """示範純 Function Call"""
    print("=== 純 Function Call 示範 ===")
    
    user_input = "台北天氣如何？"
    response = simulate_llm_with_function_call(user_input)
    print(f"用戶: {user_input}")
    print(f"回答: {response}")
    print()

def demo_mcp_framework():
    """示範 MCP 框架"""
    print("=== MCP 框架示範 ===")
    
    # 建立 MCP 伺服器
    weather_server = MCPServer("weather_service")
    weather_server.register_tool("get_weather", get_weather, "獲取天氣資訊")
    weather_server.register_resource("weather_data", {"台北": "晴天", "高雄": "多雲"})
    
    # 建立 MCP 客戶端 (模擬 LLM)
    llm_client = MCPClient("claude")
    
    # 連接流程
    connection_result = llm_client.connect_to_server(weather_server)
    print(f"連接結果: {connection_result}")
    
    # 發現資源
    available_resources = llm_client.discover_resources("weather_service")
    print(f"可用資源: {available_resources}")
    
    # 通過 MCP 調用工具
    weather_result = llm_client.call_tool("weather_service", "get_weather", location="台北")
    print(f"天氣查詢結果: {weather_result}")
    print()

def demo_comparison():
    """對比示範"""
    print("=== 路徑對比 ===")
    
    print("Function Call 路徑:")
    print("用戶問題 → LLM 判斷 → 直接調用函數 → 返回結果")
    
    print("\nMCP 路徑:")
    print("用戶問題 → LLM 判斷 → MCP 協議 → 身份驗證 → 權限檢查 → 調用工具 → 結果返回")
    print()

# ============================================
# 4. 運行示範
# ============================================

if __name__ == "__main__":
    demo_function_call_only()
    demo_mcp_framework() 
    demo_comparison()
    
    print("=== 總結 ===")
    print("Function Call ≈ Tool Call: 都是調用外部功能的能力")
    print("MCP: 是管理這些調用的標準化協議框架")
    print("關係: MCP 可以包含並管理多個 Function/Tool Calls")
```