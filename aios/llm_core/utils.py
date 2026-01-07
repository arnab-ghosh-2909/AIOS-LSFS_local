# # aios\llm_core\utils.py

# import json
# import re
# import uuid

# def tool_calling_input_format(messages: list, tools: list) -> list:
#     """
#     Integrate tool information into the messages for open-sourced LLMs.

#     Args:
#         messages (list): A list of message dictionaries, each containing at least a "role" 
#                          and "content" field. Some messages may contain "tool_calls".
#         tools (list): A list of available tool definitions, formatted as dictionaries.

#     Returns:
#         list: The updated messages list, where:
#               - Tool call messages are formatted properly for models without built-in tool support.
#               - Messages indicating tool execution results are transformed into a user message.
#               - The last message includes an instruction prompt detailing tool usage requirements.

#     Example:
#         ```python
#         messages = [
#             {"role": "user", "content": "Translate 'hello' to French."},
#             {"role": "assistant", "tool_calls": [{"name": "translate", "parameters": {"text": "hello", "language": "fr"}}]}
#         ]
        
#         tools = [{"name": "translate", "description": "Translates text into another language."}]
        
#         updated_messages = tool_calling_input_format(messages, tools)
#         print(updated_messages)
#         ```
#     """
#     prefix_prompt = (
#         "In and only in current step, you need to call tools. Available tools are: "
#     )
#     tool_prompt = json.dumps(tools)
#     suffix_prompt = "".join(
#         [
#             "Must call functions that are available. To call a function, respond "
#             "immediately and only with a list of JSON object of the following format:"
#             '[{"name":"function_name_value","parameters":{"parameter_name1":"parameter_value1",'
#             '"parameter_name2":"parameter_value2"}}]'
#         ]
#     )

#     # translate tool call message for models don't support tool call
#     for message in messages:
#         if "tool_calls" in message:
#             message["content"] = json.dumps(message.pop("tool_calls"))
            
#         elif message["role"] == "tool":
#             message["role"] = "user"
#             tool_call_id = message.pop("tool_call_id")
#             content = message.pop("content")
#             message["content"] = (
#                 f"The result of the execution of function(id :{tool_call_id}) is: {content}. "
#             )

#     messages[-1]["content"] += prefix_prompt + tool_prompt + suffix_prompt
#     return messages

# def parse_json_format(message: str) -> str:
#     """
#     Extract and parse a JSON object or array from a given string.

#     Args:
#         message (str): The input string potentially containing a JSON object or array.

#     Returns:
#         str: A string representation of the extracted JSON object or array.
    
#     Example:
#         ```python
#         message = "Here is some data: {\"key\": \"value\"}"
#         parsed_json = parse_json_format(message)
#         print(parsed_json)  # Output: '{"key": "value"}'
#         ```
#     """
#     json_array_pattern = r"\[\s*\{.*?\}\s*\]"
#     json_object_pattern = r"\{\s*.*?\s*\}"

#     match_array = re.search(json_array_pattern, message)

#     if match_array:
#         json_array_substring = match_array.group(0)

#         try:
#             json_array_data = json.loads(json_array_substring)
#             return json.dumps(json_array_data)
#         except json.JSONDecodeError:
#             pass

#     match_object = re.search(json_object_pattern, message)

#     if match_object:
#         json_object_substring = match_object.group(0)

#         try:
#             json_object_data = json.loads(json_object_substring)
#             return json.dumps(json_object_data)
#         except json.JSONDecodeError:
#             pass
#     return "[]"

# def generator_tool_call_id():
#     """
#     Generate a unique identifier for a tool call.

#     This function creates a new UUID (Universally Unique Identifier) and returns it as a string.

#     Returns:
#         str: A unique tool call ID.
    
#     Example:
#         ```python
#         tool_call_id = generator_tool_call_id()
#         print(tool_call_id)  # Example output: 'f3f2e850-b5d4-11ef-ac7e-96584d5248b2'
#         ```
#     """
#     return str(uuid.uuid4())

# def decode_litellm_tool_calls(response):
#     """
#     Decode tool call responses from LiteLLM API format.

#     Args:
#         response: The response object from LiteLLM API.

#     Returns:
#         list: A list of dictionaries, each containing:
#               - "name": The name of the function being called.
#               - "parameters": The arguments passed to the function.
#               - "id": The unique identifier of the tool call.

#     Example:
#         ```python
#         response = <LiteLLM API response>
#         decoded_calls = decode_litellm_tool_calls(response)
#         print(decoded_calls)  
#         # Output: [{'name': 'translate', 'parameters': {'text': 'hello', 'lang': 'fr'}, 'id': 'uuid1234'}]
#         ```
#     """
#     decoded_tool_calls = []
    
#     if response.choices[0].message.content is None:
#         assert response.choices[0].message.tool_calls is not None
#         tool_calls = response.choices[0].message.tool_calls

#         for tool_call in tool_calls:
#             decoded_tool_calls.append(
#                 {
#                     "name": tool_call.function.name,
#                     "parameters": tool_call.function.arguments,
#                     "id": tool_call.id
#                 }
#             )
#     else:
#         assert response.choices[0].message.content is not None
        
#         # breakpoint()
#         tool_calls = response.choices[0].message.content
#         if isinstance(tool_calls, str):
#             tool_calls = json.loads(tool_calls)
        
#         if not isinstance(tool_calls, list):
#             tool_calls = [tool_calls]
            
#         for tool_call in tool_calls:
#             decoded_tool_calls.append(
#                 {
#                     "name": tool_call["name"],
#                     "parameters": tool_call["arguments"],
#                     "id": generator_tool_call_id()
#                 }
#             )
        
#     return decoded_tool_calls

# def parse_tool_calls(message):
#     """
#     Parse and process tool calls from a message string.

#     Args:
#         message (str): A JSON string representing tool calls.

#     Returns:
#         list: A list of processed tool calls with unique IDs.

#     Example:
#         ```python
#         message = '[{"name": "text_translate", "parameters": {"text": "hello", "lang": "fr"}}]'
#         parsed_calls = parse_tool_calls(message)
#         print(parsed_calls)  
#         # Output: [{'name': 'text/translate', 'parameters': {'text': 'hello', 'lang': 'fr'}, 'id': 'uuid1234'}]
#         ```
#     """
#     # add tool call id and type for models don't support tool call
#     # if isinstance(message, dict):
#     #     message = [message]
#     # tool_calls = json.loads(parse_json_format(message))
#     tool_calls = json.loads(message)
#     # breakpoint()
#     # tool_calls = json.loads(message)
#     if isinstance(tool_calls, dict):
#         tool_calls = [tool_calls]
        
#     for tool_call in tool_calls:
#         tool_call["id"] = generator_tool_call_id()
#         # if "function" in tool_call:
        
#     tool_calls = double_underscore_to_slash(tool_calls)
#         # tool_call["type"] = "function"
#     return tool_calls

# def slash_to_double_underscore(tools):
#     """
#     Convert function names by replacing slashes ("/") with double underscores ("__").

#     Args:
#         tools (list): A list of tool dictionaries.

#     Returns:
#         list: The updated tools list with function names formatted properly.

#     Example:
#         ```python
#         tools = [{"function": {"name": "text/translate"}}]
#         formatted_tools = slash_to_double_underscore(tools)
#         print(formatted_tools)  
#         # Output: [{'function': {'name': 'text__translate'}}]
#         ```
#     """
#     for tool in tools:
#         tool_name = tool["function"]["name"]
#         if "/" in tool_name:
#             tool_name = "__".join(tool_name.split("/"))
#             tool["function"]["name"] = tool_name
#     return tools

# def double_underscore_to_slash(tool_calls):
#     """
#     Convert function names by replacing double underscores ("__") back to slashes ("/").

#     Args:
#         tool_calls (list): A list of tool call dictionaries.

#     Returns:
#         list: The updated tool calls list with function names restored to their original format.

#     Example:
#         ```python
#         tool_calls = [{"name": "text__translate", "parameters": '{"text": "hello", "lang": "fr"}'}]
#         restored_calls = double_underscore_to_slash(tool_calls)
#         print(restored_calls)  
#         # Output: [{'name': 'text/translate', 'parameters': {'text': 'hello', 'lang': 'fr'}}]
#         ```
#     """
#     for tool_call in tool_calls:
#         tool_call["name"] = tool_call["name"].replace("__", "/")
#         if isinstance(tool_call["parameters"], str):
#             tool_call["parameters"] = json.loads(tool_call["parameters"])
#         # tool_call["parameters"] = json.loads(tool_call["parameters"])
#     return tool_calls

# def pre_process_tools(tools):
#     """
#     Pre-process tool definitions by replacing slashes ("/") with double underscores ("__").

#     Args:
#         tools (list): A list of tool dictionaries.

#     Returns:
#         list: The processed tools list with modified function names.

#     Example:
#         ```python
#         tools = [{"function": {"name": "text/translate"}}]
#         preprocessed_tools = pre_process_tools(tools)
#         print(preprocessed_tools)  
#         # Output: [{'function': {'name': 'text__translate'}}]
#         ```
#     """
#     for tool in tools:
#         tool_name = tool["function"]["name"]
#         if "/" in tool_name:
#             tool_name = "__".join(tool_name.split("/"))
#             tool["function"]["name"] = tool_name
#     return tools









import json
import re
import uuid
import logging

logger = logging.getLogger(__name__)

def tool_calling_input_format(messages: list, tools: list) -> list:
    """
    Integrate tool information into the messages for open-sourced LLMs (like Krutrim/Gemma)
    that do not support native tool calling.
    """
    # 1. Format tools into a string description
    tool_desc = json.dumps(tools, indent=2)
    
    # 2. Create the system instruction
    # We are very explicit about NOT adding extra text to prevent JSON parsing errors.
    system_instruction_text = (
        "You are an AI assistant capable of using tools. "
        "You must output a JSON list of tool calls to execute actions.\n\n"
        f"AVAILABLE TOOLS:\n{tool_desc}\n\n"
        "RESPONSE FORMAT:\n"
        "You must strictly output a valid JSON list like this:\n"
        "[\n"
        "  {\n"
        "    \"name\": \"tool_name\",\n"
        "    \"arguments\": { \"arg1\": \"value1\" }\n"
        "  }\n"
        "]\n"
        "Do not add any text, explanations, or markdown code blocks before or after the JSON."
    )

    # 3. Handle existing messages
    # If the first message is system, append to it. Otherwise, insert new system message.
    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] += "\n\n" + system_instruction_text
    else:
        messages.insert(0, {"role": "system", "content": system_instruction_text})

    # 4. Handle previous tool calls in history (for conversation context)
    for message in messages:
        if "tool_calls" in message:
            # Convert previous native tool calls to text for the model to see
            message["content"] = json.dumps(message.pop("tool_calls"))
            
        elif message.get("role") == "tool":
            # Convert tool results to user messages
            message["role"] = "user"
            tool_call_id = message.pop("tool_call_id", "unknown")
            content = message.pop("content", "")
            message["content"] = f"Result of function execution (id: {tool_call_id}): {content}"

    return messages

def parse_json_format(message: str) -> str:
    """Extract and parse a JSON object or array from a given string using Regex."""
    try:
        # Regex to find JSON block wrapped in markdown or just plain
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", message, re.DOTALL)
        if match:
            return json.dumps(json.loads(match.group(1)))
        
        # Try finding raw JSON object
        start = message.find('{')
        end = message.rfind('}')
        if start != -1 and end != -1:
            return json.dumps(json.loads(message[start:end+1]))
            
        # Try finding raw JSON array
        start_arr = message.find('[')
        end_arr = message.rfind(']')
        if start_arr != -1 and end_arr != -1:
            return json.dumps(json.loads(message[start_arr:end_arr+1]))

        return message
    except Exception:
        return "[]"

def generator_tool_call_id():
    """Generate a unique identifier for a tool call."""
    return str(uuid.uuid4())

def decode_litellm_tool_calls(response):
    """Decode tool call responses from LiteLLM/OpenAI API format."""
    decoded_tool_calls = []
    
    # Check if native tool_calls exist
    message = response.choices[0].message
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            args = tool_call.function.arguments
            # Ensure arguments are a dict
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    pass
            
            decoded_tool_calls.append({
                "name": tool_call.function.name,
                "parameters": args,
                "id": tool_call.id
            })
    # Check if content has the tool call (manual mode fallback)
    elif message.content:
        content = message.content
        try:
            # Attempt to parse JSON from content
            parsed = parse_tool_calls(content)
            if parsed:
                return parsed
        except:
            pass
            
    return decoded_tool_calls

def parse_tool_calls(content: str):
    """
    Robustly extracts tool calls from model output, handling Markdown blocks 
    and extra text using Regex. This fixes the 'Expecting value' error.
    """
    if not content:
        return None

    content = content.strip()
    json_str = content

    # 1. Regex Extraction strategies
    # Strategy A: Markdown blocks ```json ... ```
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Strategy B: Find outer brackets [ ... ]
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
        else:
            # Strategy C: Find single object { ... } and wrap in list
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = "[" + content[start:end+1] + "]"

    # 2. Parsing
    try:
        tool_calls_data = json.loads(json_str)
        
        normalized_calls = []
        if isinstance(tool_calls_data, dict):
            tool_calls_data = [tool_calls_data]
            
        for tool in tool_calls_data:
            # Normalize field names
            name = tool.get("name") or tool.get("tool_name") or tool.get("function")
            # Handle nested function objects (e.g. {"function": {"name": ...}})
            if isinstance(name, dict):
                args = name.get("arguments") or name.get("parameters") or {}
                name = name.get("name")
            else:
                args = tool.get("arguments") or tool.get("parameters") or tool.get("args") or {}

            # Parse string arguments if necessary
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    pass

            if name:
                normalized_calls.append({
                    "name": name,
                    "parameters": args, # AIOS expects 'parameters' often, but sometimes 'arguments'
                    "arguments": args,  # Support both keys for safety
                    "id": generator_tool_call_id()
                })
        
        # 3. Post-process names
        return double_underscore_to_slash(normalized_calls)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool calls. Content was: {content[:50]}... Error: {e}")
        return None

def slash_to_double_underscore(tools: list) -> list:
    """Replaces slashes in tool names with double underscores."""
    if not tools: return tools
    new_tools = []
    # Handle list of tools (definitions) or list of tool calls
    for tool in tools:
        new_tool = tool.copy()
        # Case 1: Tool Definition (contains "function": {"name": ...})
        if "function" in new_tool and "name" in new_tool["function"]:
             new_tool["function"]["name"] = new_tool["function"]["name"].replace("/", "__")
        # Case 2: Tool Call (contains "name": ...)
        elif "name" in new_tool:
            new_tool["name"] = new_tool["name"].replace("/", "__")
        new_tools.append(new_tool)
    return new_tools

def double_underscore_to_slash(tool_calls: list) -> list:
    """Replaces double underscores in tool names with slashes."""
    if not tool_calls: return tool_calls
    new_calls = []
    for call in tool_calls:
        new_call = call.copy()
        if "name" in new_call:
            new_call["name"] = new_call["name"].replace("__", "/")
        new_calls.append(new_call)
    return new_calls

def pre_process_tools(tools):
    """Wrapper for slash conversion."""
    return slash_to_double_underscore(tools)

