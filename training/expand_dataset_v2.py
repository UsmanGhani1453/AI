import json
import random
from pathlib import Path


DATA_DIR = Path(r"C:\AI\training\data")

TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"


SYSTEM_PROMPT = (
    "You are a professional cold outreach and sales email assistant "
    "for a truck dispatching service. Write concise, natural, "
    "professional messages. Do not make unsupported claims. "
    "Do not use aggressive sales language."
)


def make(user, assistant):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user,
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ]
    }


# ------------------------------------------------------------
# Load the existing 35 examples
# ------------------------------------------------------------

existing = []

for file in [TRAIN_FILE, VALIDATION_FILE]:
    if file.exists():
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))


# ------------------------------------------------------------
# Additional examples
# ------------------------------------------------------------

new_examples = [

# First contact
make(
    "Write a concise first-touch email to a trucking company.",
    "Subject: Dispatching support\n\n"
    "Hi {{name}},\n\n"
    "I wanted to reach out and introduce our truck dispatching service. "
    "We help carriers with their day-to-day dispatching and load coordination.\n\n"
    "Would you be open to a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a professional cold email with a simple call to action.",
    "Subject: Quick question\n\n"
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to ask how you're currently "
    "handling dispatching for your fleet.\n\n"
    "We provide dispatching support for carriers. Would it make sense "
    "to have a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a short outreach email that does not sound like spam.",
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service and see how "
    "you're currently handling dispatching at {{company}}.\n\n"
    "Open to a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a cold email for a carrier that may need dispatching help.",
    "Subject: Dispatching question\n\n"
    "Hi {{name}},\n\n"
    "I wanted to reach out regarding dispatching support for {{company}}.\n\n"
    "We work with carriers on load searching and dispatch coordination. "
    "If you're exploring options, I'd be happy to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write an extremely concise first-touch outreach email.",
    "Hi {{name}},\n\n"
    "Do you currently handle dispatching in-house?\n\n"
    "We provide truck dispatching support for carriers and I'd be happy "
    "to explain how it works if you're interested.\n\n"
    "Best,\n{{sender_name}}"
),

# Personalization
make(
    "Personalize an outreach email for a company named {{company}}.",
    "Subject: Dispatching at {{company}}\n\n"
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to introduce our truck "
    "dispatching service.\n\n"
    "I'm curious how you're currently handling dispatching and whether "
    "outside support is something you consider.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write outreach for a carrier with several trucks.",
    "Hi {{name}},\n\n"
    "I noticed {{company}} has a multi-truck operation, so I wanted "
    "to reach out regarding dispatching support.\n\n"
    "We help carriers manage their daily dispatching workload. "
    "Would you be open to discussing your current setup?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write outreach for a small owner-operator.",
    "Subject: Dispatching support\n\n"
    "Hi {{name}},\n\n"
    "I wanted to introduce our dispatching service. We support carriers "
    "with the daily process of finding and coordinating loads.\n\n"
    "If you're currently handling dispatching yourself, I'd be happy "
    "to explain what outside support could look like.\n\n"
    "Best,\n{{sender_name}}"
),

# Follow-ups
make(
    "The prospect did not respond to the first email. Write a polite follow-up.",
    "Hi {{name}},\n\n"
    "Just following up on my previous message regarding dispatching support.\n\n"
    "If this isn't something you're considering right now, no problem. "
    "If it is, I'm happy to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a very short follow-up after no response.",
    "Hi {{name}},\n\n"
    "Just checking back on my previous note. Would a conversation about "
    "dispatching support be relevant for {{company}}?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a final low-pressure follow-up.",
    "Hi {{name}},\n\n"
    "I'll make this my last follow-up. If dispatching support isn't a "
    "priority right now, completely understood.\n\n"
    "If that changes, I'm happy to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a follow-up that acknowledges the prospect may be busy.",
    "Hi {{name}},\n\n"
    "I know things can get busy, so I wanted to briefly follow up on "
    "my previous message.\n\n"
    "If you'd like to discuss dispatching support, I'm available.\n\n"
    "Best,\n{{sender_name}}"
),

# Interested
make(
    "The prospect says: 'I'm interested.' Respond.",
    "Thanks for getting back to me. I'd be happy to explain how our "
    "truck dispatching service works.\n\n"
    "If you tell me a little about your fleet and equipment, I can "
    "give you a more relevant overview."
),

make(
    "The prospect says: 'Tell me more.' Respond professionally.",
    "Absolutely. We provide truck dispatching support for carriers, "
    "including assistance with load searching and dispatch coordination.\n\n"
    "I'd be happy to discuss your current setup and see whether our "
    "service could be useful."
),

make(
    "The prospect says: 'I'm interested but want to know how it works.'",
    "Sure. We first understand your equipment, fleet, lanes, and "
    "preferences. From there, we can support the load-search and "
    "dispatch coordination process based on your requirements."
),

make(
    "The prospect asks whether you can discuss the service over a call.",
    "Absolutely. I'd be happy to discuss the service and answer your "
    "questions.\n\n"
    "What day and time works best for you?"
),

# Pricing
make(
    "The prospect asks for pricing. Do not invent a price.",
    "Happy to discuss pricing. The cost depends on the type of "
    "dispatching support and your operation.\n\n"
    "If you share a few details about your fleet, I can give you "
    "more relevant information."
),

make(
    "The prospect asks: 'Is there a setup fee?' Do not invent an answer.",
    "I don't want to give you inaccurate information without knowing "
    "which service setup you're considering.\n\n"
    "I can explain the applicable pricing and fees once I understand "
    "your requirements."
),

make(
    "The prospect says the price is higher than expected.",
    "I understand. The service needs to make sense financially for "
    "your operation.\n\n"
    "If you'd like, we can discuss your current setup and see whether "
    "there's a practical fit."
),

# Existing provider
make(
    "The prospect says: 'We already use a dispatch service.'",
    "Understood. If your current dispatching setup is working well, "
    "there's no need to change it.\n\n"
    "If you ever need additional support or another option in the "
    "future, feel free to reach out."
),

make(
    "The prospect says: 'We already have a dispatcher.'",
    "No problem. I appreciate you letting me know.\n\n"
    "If your situation changes or you ever need backup dispatching "
    "support, I'd be happy to connect."
),

make(
    "The prospect says they are satisfied with their current dispatcher.",
    "That's good to hear. If your current setup is working well, "
    "I completely understand staying with it.\n\n"
    "I'll leave the door open if you ever need another option."
),

# Not interested
make(
    "The prospect says: 'Not interested.'",
    "Understood. Thanks for getting back to me, {{name}}. "
    "I appreciate your time.\n\n"
    "Best wishes."
),

make(
    "The prospect says: 'Please don't contact me again.'",
    "Understood. I'll respect your request and won't follow up again.\n\n"
    "Thank you for letting me know."
),

make(
    "The prospect says they are not looking for dispatching right now.",
    "Understood. Thanks for letting me know.\n\n"
    "If your needs change in the future, you're welcome to reach out.\n\n"
    "Best,\n{{sender_name}}"
),

# Questions
make(
    "The prospect asks: 'What equipment do you work with?' Avoid unsupported claims.",
    "We can determine whether our service is suitable based on your "
    "equipment and operation.\n\n"
    "If you tell me what type of equipment you're running, I can "
    "discuss the fit with you."
),

make(
    "The prospect asks: 'Do you handle load searching?'",
    "Yes, load searching can be part of the dispatching support we "
    "provide. We can discuss your preferred equipment, lanes, and "
    "requirements to understand what you need."
),

make(
    "The prospect asks how dispatching support can help them.",
    "Dispatching support can help manage parts of the daily process "
    "such as finding suitable loads, coordinating load details, and "
    "keeping dispatch communication organized."
),

make(
    "The prospect asks what information you need from them.",
    "I'd need some basic information about your operation, such as "
    "your equipment type, fleet size, preferred lanes, and dispatching "
    "requirements. That helps us understand whether the service is a fit."
),

# Calls
make(
    "The prospect asks for a phone call.",
    "Sure. I'd be happy to speak with you.\n\n"
    "What time works best for you?"
),

make(
    "The prospect says: 'Call me this afternoon.'",
    "Absolutely. What time this afternoon works best for you?"
),

make(
    "The prospect wants to schedule a call next week.",
    "Sounds good. Let me know which day and time next week is convenient "
    "for you, and we can arrange the conversation."
),

# Busy
make(
    "The prospect says: 'I'm busy right now.'",
    "No problem. I understand.\n\n"
    "If there's a better time to connect, just let me know."
),

make(
    "The prospect says: 'I don't have time to talk.'",
    "Understood. I'll keep it brief and won't take more of your time.\n\n"
    "If dispatching support becomes relevant later, feel free to reach out."
),

# Information
make(
    "The prospect asks you to send information by email.",
    "Of course. I'll keep the information concise and focused on our "
    "truck dispatching service.\n\n"
    "If you share your fleet size and equipment type, I can make it "
    "more relevant to your operation."
),

make(
    "The prospect says: 'Just send me your company information.'",
    "Absolutely. I'll send over a concise overview of our truck "
    "dispatching service and the type of support we provide."
),

# Objections
make(
    "The prospect says: 'We do our own dispatching.'",
    "Understood. If handling dispatching internally is working well "
    "for your operation, that makes sense.\n\n"
    "If you ever want to compare your current process with outside "
    "support, I'd be happy to connect."
),

make(
    "The prospect says: 'I don't trust dispatching companies.'",
    "I understand the concern. Choosing a dispatching service is "
    "something you should evaluate carefully.\n\n"
    "I'm happy to answer your questions about how our service works "
    "so you can decide whether it's relevant."
),

make(
    "The prospect says: 'We've had a bad experience with dispatchers.'",
    "I understand why that would make you cautious.\n\n"
    "If you're open to it, I can explain how our dispatching process "
    "works and answer any questions you have."
),

make(
    "The prospect says: 'I don't want to switch providers.'",
    "Understood. If you're happy with your current provider, there's "
    "no pressure to change.\n\n"
    "I'll be happy to connect if your needs change in the future."
),

# Tone rewriting
make(
    "Rewrite this aggressive sentence to sound professional: "
    "'You need our dispatching service if you want to make more money.'",
    "You may find our dispatching support useful if you're looking "
    "for assistance with your current operation."
),

make(
    "Rewrite this email to sound less sales-focused.",
    "Hi {{name}},\n\n"
    "I wanted to reach out and learn how you're currently handling "
    "dispatching at {{company}}.\n\n"
    "We provide dispatching support for carriers, and I thought it "
    "would be useful to introduce ourselves.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Shorten a long truck dispatching outreach email.",
    "Subject: Quick question\n\n"
    "Hi {{name}},\n\n"
    "How are you currently handling dispatching at {{company}}?\n\n"
    "We provide truck dispatching support and would be happy to "
    "connect if you're exploring options.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Rewrite a generic outreach email to sound more conversational.",
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to reach out. I'm curious "
    "how you're currently handling dispatching for your fleet.\n\n"
    "We provide dispatching support for carriers and I'd be happy "
    "to compare notes if useful.\n\n"
    "Best,\n{{sender_name}}"
),

# Feedback learning
make(
    "Feedback: The email is too long. Make it shorter.",
    "Hi {{name}},\n\n"
    "I wanted to reach out regarding dispatching support for {{company}}.\n\n"
    "How are you currently handling dispatching for your fleet?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Feedback: The email sounds robotic. Make it natural.",
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to introduce myself. "
    "I'm curious how you're currently handling dispatching.\n\n"
    "We provide dispatching support for carriers if you ever want "
    "to explore another option.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Feedback: Remove the exaggerated claim that we are the best.",
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service. We support "
    "carriers with their day-to-day dispatching and load coordination.\n\n"
    "Would you be open to a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Feedback: The call to action is too aggressive. Make it softer.",
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to introduce our dispatching "
    "service.\n\n"
    "If this is relevant to your operation, I'd be happy to have a "
    "quick conversation.\n\n"
    "Best,\n{{sender_name}}"
),

# Conversation handling
make(
    "The prospect asks: 'Where did you find my company?'",
    "I came across {{company}} while researching trucking companies "
    "and wanted to introduce our dispatching service."
),

make(
    "The prospect asks: 'Why are you contacting me?'",
    "I reached out because {{company}} is a trucking operation and "
    "we provide dispatching support for carriers.\n\n"
    "I wanted to see how you're currently handling dispatching and "
    "whether an outside option might be relevant."
),

make(
    "The prospect asks: 'Are you a broker?'",
    "We're offering truck dispatching support rather than approaching "
    "you as a freight broker. I'd be happy to explain our service "
    "and how the dispatching relationship works."
),

make(
    "The prospect says: 'I'm interested, but I need to think about it.'",
    "Of course. Take your time.\n\n"
    "If you have any questions while considering it, feel free to "
    "send them over and I'll be happy to answer."
),

make(
    "The prospect says: 'Maybe later.'",
    "Understood. Timing matters.\n\n"
    "I'll leave the door open, and if dispatching support becomes "
    "relevant later, you're welcome to reach out."
),

make(
    "The prospect asks for references or proof of service.",
    "I understand. It's reasonable to evaluate a service before "
    "moving forward.\n\n"
    "I can provide information that we actually have available "
    "and answer any specific questions about our process."
),

make(
    "The prospect asks for guaranteed results. Do not make guarantees.",
    "I wouldn't want to promise a guaranteed result because every "
    "operation is different.\n\n"
    "What I can do is explain our dispatching process and what type "
    "of support we provide so you can evaluate whether it fits."
),

make(
    "The prospect gives a positive response but does not ask a question.",
    "Thanks, {{name}}. I appreciate the response.\n\n"
    "If you're open to it, we can have a quick conversation about "
    "your current dispatching setup and what you're looking for."
),

make(
    "The prospect says they want to start immediately.",
    "Great. Before getting started, I'd like to understand your fleet, "
    "equipment, lanes, and dispatching requirements so we can confirm "
    "the setup and next steps."
),

make(
    "The prospect asks what the next step is.",
    "The next step would be to understand your fleet and dispatching "
    "requirements, then discuss whether our service is a suitable fit."
),

# Email quality
make(
    "Write an outreach email with no exaggerated claims.",
    "Subject: Dispatching support\n\n"
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service. We help "
    "carriers with load searching and dispatch coordination.\n\n"
    "If you're open to discussing your current setup, I'd be happy "
    "to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a cold email using a question as the opening.",
    "Subject: Quick question\n\n"
    "Hi {{name}},\n\n"
    "How are you currently handling dispatching for {{company}}?\n\n"
    "We provide truck dispatching support for carriers, and I'd be "
    "happy to explain our service if you're exploring options.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a professional email that avoids excessive marketing language.",
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service and learn "
    "a little about your current dispatching process.\n\n"
    "If outside support is something you're considering, I'd be happy "
    "to have a conversation.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a short outreach email with one clear call to action.",
    "Subject: Dispatching question\n\n"
    "Hi {{name}},\n\n"
    "We provide truck dispatching support for carriers. "
    "How are you currently handling dispatching at {{company}}?\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write an email that respects a prospect who is already satisfied.",
    "Hi {{name}},\n\n"
    "I understand if you're happy with your current dispatching setup. "
    "I just wanted to introduce our service in case you ever need "
    "another option.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a polite response when the prospect declines.",
    "Thanks for letting me know, {{name}}. I appreciate the response "
    "and respect your decision.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a short response when the prospect asks you to stop following up.",
    "Understood. I respect your request and won't follow up again.\n\n"
    "Thank you for letting me know."
),

make(
    "Write a follow-up that adds no new pressure.",
    "Hi {{name}},\n\n"
    "Just checking back on my earlier message. If dispatching support "
    "isn't relevant right now, completely understood.\n\n"
    "Best,\n{{sender_name}}"
),

make(
    "Write a response to a prospect who asks for a few days to decide.",
    "Of course. Take the time you need.\n\n"
    "If any questions come up while you're deciding, feel free to "
    "send them over."
),

make(
    "Write a professional response to a prospect who asks a question "
    "you cannot answer without more information.",
    "I'd be happy to give you an accurate answer. I just need a little "
    "more information about your fleet and current setup first."
),
]


# ------------------------------------------------------------
# Combine and validate
# ------------------------------------------------------------

all_examples = existing + new_examples

# Remove exact duplicate JSON examples
unique = []
seen = set()

for item in all_examples:
    key = json.dumps(item, sort_keys=True)
    if key not in seen:
        seen.add(key)
        unique.append(item)

print("Existing examples:", len(existing))
print("New examples:", len(new_examples))
print("Unique examples:", len(unique))


if len(unique) < 100:
    raise RuntimeError(
        f"Only {len(unique)} unique examples available. "
        "Need at least 100."
    )


# Exactly 100 examples
unique = unique[:100]

random.seed(42)
random.shuffle(unique)

train_examples = unique[:80]
validation_examples = unique[80:100]


# ------------------------------------------------------------
# Save final dataset
# ------------------------------------------------------------

with TRAIN_FILE.open("w", encoding="utf-8") as f:
    for item in train_examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


with VALIDATION_FILE.open("w", encoding="utf-8") as f:
    for item in validation_examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


print()
print("======================================")
print("VERSION 2 DATASET READY")
print("======================================")
print("Training examples:   ", len(train_examples))
print("Validation examples: ", len(validation_examples))
print("Total:                ", len(train_examples) + len(validation_examples))
print()
print("Train:", TRAIN_FILE)
print("Validation:", VALIDATION_FILE)