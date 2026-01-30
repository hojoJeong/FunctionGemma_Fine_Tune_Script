from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json


# Prompts for mobile-action test
system_prompt =  "Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2025-07-10T19:06:29\nDay of week is Thursday\nYou are a model that can do function calling with the following functions\n"
user_prompt = 'Schedule a "team meeting" tomorrow at 4pm.'

# Model version, path
model_version = "v1"
model_path = f""

# Load model and tokenizer
model_id = "jprtr/google_mobile_actions"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="eager",
    torch_dtype="auto",
)

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the tools (function schemas)
tools = [
    {
        "function": {
            "name": "create_calendar_event",
            "description": "Creates a new calendar event.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING", "description": "The title of the event."},
                    "datetime": {"type": "STRING", "description": "The date and time in YYYY-MM-DDTHH:MM:SS format."},
                },
                "required": ["title", "datetime"],
            },
        }
    },
    {
        "function": {
            "name": "send_email",
            "description": "Sends an email.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "to": {"type": "STRING", "description": "The recipient email address."},
                    "subject": {"type": "STRING", "description": "The email subject."},
                    "body": {"type": "STRING", "description": "The email body."},
                },
                "required": ["to", "subject"],
            },
        }
    },
    # ... add other function definitions
]

# Create messages
messages = [
    {
        "role": "developer",
        "content": (
           system_prompt
        ),
    },
    {
        "role": "user",
        "content": user_prompt,
    },
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
)

# Generate
output = pipe(prompt, max_new_tokens=200)[0]["generated_text"][len(prompt):].strip()

# Save model
model.save_pretrained(model_path)

# Save tokenizer
tokenizer.save_pretrained(model_path)

print(f"\n\033[1m#####\nModel output: {output}\nModel Path : {model_path}\nModel Version : {model_version}\n#####\033[0m")
# Example output: call:create_calendar_event{datetime:2025-07-11T16:00:00,title:team meeting}