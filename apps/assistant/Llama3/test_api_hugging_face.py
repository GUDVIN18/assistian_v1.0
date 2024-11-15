from hugchat import hugchat
from hugchat.login import Login

# Учетные данные для входа в Hugging Face
EMAIL = "ananjev.05@gmail.com"
PASSWD = "20060228Dima"
cookie_path_dir = "./cookies/" # NOTE: trailing slash (/) is required to avoid errors
sign = Login(EMAIL, PASSWD)
cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

# Create your ChatBot
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"

message_result = chatbot.chat("Привет!") # note: message_result is a generator, the method will return immediately.

# Non stream
message_str: str = message_result.wait_until_done() # you can also print(message_result) directly. 
# get files(such as images)
file_list = message_result.get_files_created() # must call wait_until_done() first!
print(file_list)
# tips: model "CohereForAI/c4ai-command-r-plus" can generate images :)







# # Stream response
# for resp in chatbot.chat(
#     "Hello",
#     stream=True
# ):
#     print(resp)

# # Web search
# query_result = chatbot.chat("Hi!", web_search=True)
# print(query_result)
# for source in query_result.web_search_sources:
#     print(source.link)
#     print(source.title)

# # Create a new conversation
# chatbot.new_conversation(switch_to = True) # switch to the new conversation

# # Get conversations on the server that are not from the current session (all your conversations in huggingchat)
# conversation_list = chatbot.get_remote_conversations(replace_conversation_list=True)
# # Get conversation list(local)
# conversation_list = chatbot.get_conversation_list()

# # Get the available models (not hardcore)
# models = chatbot.get_available_llm_models()

# # Switch model with given index
# chatbot.switch_llm(0) # Switch to the first model
# chatbot.switch_llm(1) # Switch to the second model

# # Get information about the current conversation
# info = chatbot.get_conversation_info()
# print(info.id, info.title, info.model, info.system_prompt, info.history)

# # Assistant
# assistant = chatbot.search_assistant(assistant_name="ChatGpt") # assistant name list in https://huggingface.co/chat/assistants
# assistant_list = chatbot.get_assistant_list_by_page(page=0)
# chatbot.new_conversation(assistant=assistant, switch_to=True) # create a new conversation with assistant

# # [DANGER] Delete all the conversations for the logged in user
# chatbot.delete_all_conversations()