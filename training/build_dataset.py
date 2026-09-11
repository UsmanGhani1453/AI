"""
build_dataset.py

Generates the fine-tuning dataset used to train the truck-dispatching
correspondence assistant.

The dataset covers two related but distinct writing tasks, each with its
own system prompt:

    1. General business correspondence  (follow-ups, scheduling, account
       management, referrals, apologies, rate confirmations, billing,
       compliance, driver dispatch, escalations, renewals, short-form
       replies, etc.)
    2. Cold outreach / sales email writing (first contact, objection
       handling, tone rewriting, lane/backhaul outreach, etc.)

Every example is a single-turn ``{system, user, assistant}`` conversation
in OpenAI/HuggingFace chat format, written in polished, formal-professional
business English: correct grammar and punctuation, consistent formatting,
no slang, no unsupported claims, and no aggressive sales language.

Running this script deterministically regenerates:

    training/data/train.jsonl        (111 examples)
    training/data/validation.jsonl   (28 examples)

Usage:
    python training/build_dataset.py
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"

MIN_EXAMPLES = 139
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

SYSTEM_PROMPT_GENERAL = (
    "You are a professional business correspondence assistant for a "
    "truck dispatching service. Write clear, courteous, and fully "
    "professional emails and responses. Use correct grammar and "
    "punctuation at all times. Do not make unsupported claims. Do "
    "not use aggressive sales language, slang, or casual phrasing."
)

SYSTEM_PROMPT_OUTREACH = (
    "You are a professional cold outreach and sales email assistant "
    "for a truck dispatching service. Write concise, natural, "
    "professional messages. Do not make unsupported claims. "
    "Do not use aggressive sales language."
)


def make_general(user, assistant):
    """Build a chat example using the general correspondence persona."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_GENERAL},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def make_outreach(user, assistant):
    """Build a chat example using the cold-outreach persona."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_OUTREACH},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ================================================================
# PART 1 — General business correspondence (7 categories)
# ================================================================

examples = []

# ============================================================
# 1. FIRST CONTACT / INTRODUCTION
# ============================================================

examples += [
    make_general(
        "Write a professional first-contact email introducing our "
        "truck dispatching service to a carrier.",
        "Subject: Introduction — Dispatching Support for {{company}}\n\n"
        "Dear {{name}},\n\n"
        "My name is {{sender_name}}, and I am reaching out on behalf "
        "of our truck dispatching service. We assist carriers with "
        "load searching, rate negotiation, and day-to-day dispatch "
        "coordination.\n\n"
        "I would welcome the opportunity to learn more about how "
        "{{company}} currently manages dispatching and to determine "
        "whether our service may be of value to your operation.\n\n"
        "Please let me know if you would be open to a brief "
        "conversation at your convenience.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a concise, professional introductory email for a "
        "trucking company.",
        "Subject: Dispatching Support Introduction\n\n"
        "Dear {{name}},\n\n"
        "I am writing to introduce our dispatching service, which "
        "supports carriers with load coordination and daily "
        "operational planning.\n\n"
        "If it would be helpful, I would be glad to schedule a short "
        "call to discuss your current dispatching process and answer "
        "any questions you may have.\n\n"
        "Sincerely,\n{{sender_name}}",
    ),
    make_general(
        "Write a formal outreach email that opens a conversation "
        "rather than making a direct sales pitch.",
        "Subject: A Question About Your Dispatching Process\n\n"
        "Dear {{name}},\n\n"
        "I hope this message finds you well. I am reaching out to "
        "learn more about how {{company}} currently manages "
        "dispatching.\n\n"
        "Our company provides dispatching support for carriers, and "
        "I would appreciate the opportunity to discuss your current "
        "approach and share how we might be able to assist.\n\n"
        "Thank you for your time, and I look forward to your reply.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional introduction email for an "
        "owner-operator with a single truck.",
        "Subject: Introduction — Dispatching Support\n\n"
        "Dear {{name}},\n\n"
        "I understand that managing dispatching independently can be "
        "demanding, particularly for an owner-operator. I am writing "
        "to introduce our dispatching service, which handles load "
        "searching and coordination on behalf of our clients.\n\n"
        "I would be happy to discuss how this service could support "
        "your operation, at a time that is convenient for you.\n\n"
        "Kind regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional email introducing our service to a "
        "fleet manager overseeing multiple trucks.",
        "Subject: Dispatching Support for Multi-Truck Fleets\n\n"
        "Dear {{name}},\n\n"
        "I am writing to introduce our dispatching service, which is "
        "designed to support fleet operations of varying sizes with "
        "load coordination and daily dispatch management.\n\n"
        "I would welcome the opportunity to discuss {{company}}'s "
        "current dispatching structure and identify areas where our "
        "service may be beneficial.\n\n"
        "Please let me know a convenient time to speak.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
]

# ============================================================
# 2. FOLLOW-UP EMAILS
# ============================================================

examples += [
    make_general(
        "The prospect has not responded to the first email. Write a "
        "polite, professional follow-up.",
        "Subject: Following Up — Dispatching Support\n\n"
        "Dear {{name}},\n\n"
        "I wanted to follow up on my previous message regarding "
        "dispatching support for {{company}}. I understand you may "
        "have a busy schedule, and I did not want my note to go "
        "unnoticed.\n\n"
        "If this is something you would like to discuss further, "
        "please let me know a convenient time. If not, I completely "
        "understand.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a brief, professional second follow-up after no reply.",
        "Subject: Checking In\n\n"
        "Dear {{name}},\n\n"
        "I am following up once more regarding dispatching support "
        "for {{company}}. Please let me know if this would be a "
        "relevant conversation to have at this time.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a final, respectful follow-up email that does not "
        "pressure the recipient.",
        "Subject: Final Follow-Up\n\n"
        "Dear {{name}},\n\n"
        "This will be my final follow-up regarding dispatching "
        "support. If this is not a priority for {{company}} at "
        "present, I fully understand.\n\n"
        "Should your needs change in the future, I would be glad to "
        "reconnect.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a follow-up email acknowledging that the recipient may "
        "have a demanding schedule.",
        "Subject: Re: Dispatching Support\n\n"
        "Dear {{name}},\n\n"
        "I recognize that schedules in this industry can be "
        "demanding, so I wanted to briefly follow up on my earlier "
        "message.\n\n"
        "If a short conversation about dispatching support would be "
        "useful, I am available at your convenience.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a follow-up referencing a prior conversation at an "
        "industry event.",
        "Subject: Following Up From {{event_name}}\n\n"
        "Dear {{name}},\n\n"
        "It was a pleasure speaking with you at {{event_name}}. As "
        "discussed, I wanted to follow up regarding dispatching "
        "support for {{company}}.\n\n"
        "Please let me know if you would like to continue the "
        "conversation, and I will arrange a time that works well for "
        "you.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
]

# ============================================================
# 3. RESPONSES TO INTERESTED PROSPECTS
# ============================================================

examples += [
    make_general(
        "The prospect replies: 'I'm interested.' Write a professional "
        "response.",
        "Dear {{name}},\n\n"
        "Thank you for your reply. I would be glad to explain how our "
        "dispatching service works.\n\n"
        "Could you share some details about your fleet and equipment? "
        "This will allow me to provide information most relevant to "
        "your operation.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "The prospect replies: 'Tell me more about your service.' "
        "Write a professional response.",
        "Dear {{name}},\n\n"
        "Certainly. Our dispatching service supports carriers with "
        "load searching, rate negotiation, and coordination of daily "
        "dispatch activities.\n\n"
        "I would be happy to discuss your current setup in more "
        "detail and outline how our service could be tailored to "
        "your needs.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "The prospect asks how the onboarding process works. Write a "
        "professional response.",
        "Dear {{name}},\n\n"
        "Thank you for your interest. Onboarding typically begins "
        "with a short discussion about your fleet, preferred lanes, "
        "and operational requirements. From there, we outline the "
        "dispatching process and align it with your expectations.\n\n"
        "I would be glad to schedule a call to walk through the "
        "details at a time convenient for you.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "The prospect asks for pricing information. Write a "
        "professional response that does not overstate details you "
        "do not have.",
        "Dear {{name}},\n\n"
        "Thank you for asking. Pricing depends on several factors, "
        "including fleet size and service scope. I would be glad to "
        "provide accurate details during a brief call, so that the "
        "information reflects your specific situation.\n\n"
        "Please let me know a time that works well for you.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "The prospect asks for a phone call. Write a short, "
        "professional response confirming availability.",
        "Dear {{name}},\n\n"
        "I would be happy to speak with you. Please let me know a "
        "time that is convenient, and I will make myself available "
        "accordingly.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
]

# ============================================================
# 4. OBJECTIONS AND DECLINES
# ============================================================

examples += [
    make_general(
        "The prospect says they already have an in-house dispatcher. "
        "Write a professional, respectful response.",
        "Dear {{name}},\n\n"
        "Thank you for letting me know. It is good to hear that "
        "{{company}} has dispatching handled internally.\n\n"
        "If your needs change in the future, or if additional support "
        "during busy periods would be helpful, please feel free to "
        "reach out.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "The prospect says they are not interested at this time. "
        "Write a professional and courteous response.",
        "Dear {{name}},\n\n"
        "Thank you for letting me know, and I appreciate your "
        "response. I understand this may not be the right time.\n\n"
        "Should your situation change, I would welcome the "
        "opportunity to reconnect.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "The prospect asks to be removed from future outreach. Write "
        "a professional response confirming the request.",
        "Dear {{name}},\n\n"
        "Thank you for letting me know. I have noted your request and "
        "will remove {{company}} from future outreach.\n\n"
        "I wish you continued success with your operations.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "The prospect expresses concern about switching from their "
        "current provider. Write a professional, reassuring but "
        "honest response.",
        "Dear {{name}},\n\n"
        "That is a reasonable concern, and I appreciate you sharing "
        "it. Transitioning dispatching providers is a significant "
        "decision, and I would not want to encourage a change without "
        "understanding your current arrangement.\n\n"
        "If it would help, I am glad to answer specific questions so "
        "that you can make an informed decision.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
]

# ============================================================
# 5. SCHEDULING AND LOGISTICS
# ============================================================

examples += [
    make_general(
        "Write a professional email proposing a specific meeting "
        "time.",
        "Subject: Proposed Meeting Time\n\n"
        "Dear {{name}},\n\n"
        "Would {{proposed_time}} work for a brief call to discuss "
        "dispatching support for {{company}}? If this time is not "
        "convenient, please suggest an alternative that suits your "
        "schedule.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a confirmation email for a scheduled call.",
        "Subject: Confirming Our Call — {{proposed_time}}\n\n"
        "Dear {{name}},\n\n"
        "I am writing to confirm our call scheduled for "
        "{{proposed_time}}. Please let me know if you need to "
        "reschedule, and I will be glad to accommodate.\n\n"
        "I look forward to speaking with you.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional email requesting to reschedule a call.",
        "Subject: Request to Reschedule\n\n"
        "Dear {{name}},\n\n"
        "I apologize for the inconvenience, but I need to reschedule "
        "our upcoming call. Could you let me know a few times that "
        "would work well for you this week?\n\n"
        "Thank you for your understanding.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional thank-you email after a call with a "
        "prospect.",
        "Subject: Thank You for Your Time\n\n"
        "Dear {{name}},\n\n"
        "Thank you for taking the time to speak with me today. It was "
        "a pleasure learning more about {{company}} and your current "
        "dispatching process.\n\n"
        "As discussed, I will follow up with the additional "
        "information you requested. Please do not hesitate to reach "
        "out with any questions in the meantime.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
]

# ============================================================
# 6. OPERATIONAL / ACCOUNT CORRESPONDENCE
# ============================================================

examples += [
    make_general(
        "Write a professional email confirming a new load assignment "
        "to a driver or carrier.",
        "Subject: Load Confirmation — {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "This message confirms the assignment of load {{load_id}}, "
        "picking up at {{pickup_location}} and delivering to "
        "{{delivery_location}}.\n\n"
        "Please review the details and let me know if you have any "
        "questions prior to pickup.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional email requesting updated availability "
        "for the upcoming week.",
        "Subject: Availability Request — Upcoming Week\n\n"
        "Dear {{name}},\n\n"
        "Could you please share your availability for the upcoming "
        "week? This will help ensure we identify suitable loads that "
        "align with your schedule and preferred lanes.\n\n"
        "Thank you for the update.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional email notifying a carrier of a delay in "
        "a scheduled pickup.",
        "Subject: Pickup Delay Notice — {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "I am writing to inform you of a delay affecting the pickup "
        "for load {{load_id}}. The revised pickup time is "
        "{{revised_time}}.\n\n"
        "I apologize for any inconvenience and will keep you updated "
        "should any further changes occur.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional email requesting rate confirmation "
        "documentation from a broker.",
        "Subject: Rate Confirmation Request — {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "Could you please send the signed rate confirmation for load "
        "{{load_id}} at your earliest convenience? This will allow us "
        "to finalize the arrangements ahead of pickup.\n\n"
        "Thank you for your assistance.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional email addressing a billing discrepancy "
        "with a client.",
        "Subject: Billing Discrepancy — {{invoice_id}}\n\n"
        "Dear {{name}},\n\n"
        "I am writing regarding invoice {{invoice_id}}, where I "
        "noticed a discrepancy that I would like to clarify. Could "
        "you please confirm the correct amount so that we can resolve "
        "this promptly?\n\n"
        "Thank you for your attention to this matter.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional email apologizing for a service issue "
        "and outlining the corrective action.",
        "Subject: Apology and Resolution — {{issue_reference}}\n\n"
        "Dear {{name}},\n\n"
        "I want to sincerely apologize for the issue you experienced "
        "regarding {{issue_reference}}. We take this seriously and "
        "have taken the following corrective action: "
        "{{corrective_action}}.\n\n"
        "Please let me know if there is anything further I can do to "
        "address this.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
]

# ============================================================
# 7. REFERRALS AND RELATIONSHIP MAINTENANCE
# ============================================================

examples += [
    make_general(
        "Write a professional email thanking a client for a referral.",
        "Subject: Thank You for Your Referral\n\n"
        "Dear {{name}},\n\n"
        "Thank you for referring {{referral_company}} to our "
        "dispatching service. Your recommendation means a great deal "
        "to us, and we will ensure the same level of attentive "
        "service is provided.\n\n"
        "Please let me know if there is ever anything further we can "
        "do for you.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional check-in email for a long-standing "
        "client, without a specific sales purpose.",
        "Subject: Checking In\n\n"
        "Dear {{name}},\n\n"
        "I wanted to check in and see how things have been going with "
        "{{company}}'s operations. Please let me know if there is "
        "anything we can assist with or improve upon on our end.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make_general(
        "Write a professional year-end thank-you email to a long-term "
        "client.",
        "Subject: Thank You for a Great Year\n\n"
        "Dear {{name}},\n\n"
        "As the year comes to a close, I wanted to take a moment to "
        "thank you for your continued partnership with our "
        "dispatching service. We value the opportunity to support "
        "{{company}}'s operations.\n\n"
        "Wishing you a successful year ahead.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
]


# ================================================================
# PART 2 — Cold outreach / sales correspondence (14 categories)
# ================================================================

new_examples = [

# First contact
make_outreach(
    "Write a concise first-touch email to a trucking company.",
    "Subject: Dispatching support\n\n"
    "Hi {{name}},\n\n"
    "I wanted to reach out and introduce our truck dispatching service. "
    "We help carriers with their day-to-day dispatching and load coordination.\n\n"
    "Would you be open to a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a professional cold email with a simple call to action.",
    "Subject: Quick question\n\n"
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to ask how you're currently "
    "handling dispatching for your fleet.\n\n"
    "We provide dispatching support for carriers. Would it make sense "
    "to have a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a short outreach email that does not sound like spam.",
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service and see how "
    "you're currently handling dispatching at {{company}}.\n\n"
    "Open to a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a cold email for a carrier that may need dispatching help.",
    "Subject: Dispatching question\n\n"
    "Hi {{name}},\n\n"
    "I wanted to reach out regarding dispatching support for {{company}}.\n\n"
    "We work with carriers on load searching and dispatch coordination. "
    "If you're exploring options, I'd be happy to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write an extremely concise first-touch outreach email.",
    "Hi {{name}},\n\n"
    "Do you currently handle dispatching in-house?\n\n"
    "We provide truck dispatching support for carriers and I'd be happy "
    "to explain how it works if you're interested.\n\n"
    "Best,\n{{sender_name}}"
),

# Personalization
make_outreach(
    "Personalize an outreach email for a company named {{company}}.",
    "Subject: Dispatching at {{company}}\n\n"
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to introduce our truck "
    "dispatching service.\n\n"
    "I'm curious how you're currently handling dispatching and whether "
    "outside support is something you consider.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write outreach for a carrier with several trucks.",
    "Hi {{name}},\n\n"
    "I noticed {{company}} has a multi-truck operation, so I wanted "
    "to reach out regarding dispatching support.\n\n"
    "We help carriers manage their daily dispatching workload. "
    "Would you be open to discussing your current setup?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
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
make_outreach(
    "The prospect did not respond to the first email. Write a polite follow-up.",
    "Hi {{name}},\n\n"
    "Just following up on my previous message regarding dispatching support.\n\n"
    "If this isn't something you're considering right now, no problem. "
    "If it is, I'm happy to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a very short follow-up after no response.",
    "Hi {{name}},\n\n"
    "Just checking back on my previous note. Would a conversation about "
    "dispatching support be relevant for {{company}}?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a final low-pressure follow-up.",
    "Hi {{name}},\n\n"
    "I'll make this my last follow-up. If dispatching support isn't a "
    "priority right now, completely understood.\n\n"
    "If that changes, I'm happy to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a follow-up that acknowledges the prospect may be busy.",
    "Hi {{name}},\n\n"
    "I know things can get busy, so I wanted to briefly follow up on "
    "my previous message.\n\n"
    "If you'd like to discuss dispatching support, I'm available.\n\n"
    "Best,\n{{sender_name}}"
),

# Interested
make_outreach(
    "The prospect says: 'I'm interested.' Respond.",
    "Thanks for getting back to me. I'd be happy to explain how our "
    "truck dispatching service works.\n\n"
    "If you tell me a little about your fleet and equipment, I can "
    "give you a more relevant overview."
),

make_outreach(
    "The prospect says: 'Tell me more.' Respond professionally.",
    "Absolutely. We provide truck dispatching support for carriers, "
    "including assistance with load searching and dispatch coordination.\n\n"
    "I'd be happy to discuss your current setup and see whether our "
    "service could be useful."
),

make_outreach(
    "The prospect says: 'I'm interested but want to know how it works.'",
    "Sure. We first understand your equipment, fleet, lanes, and "
    "preferences. From there, we can support the load-search and "
    "dispatch coordination process based on your requirements."
),

make_outreach(
    "The prospect asks whether you can discuss the service over a call.",
    "Absolutely. I'd be happy to discuss the service and answer your "
    "questions.\n\n"
    "What day and time works best for you?"
),

# Pricing
make_outreach(
    "The prospect asks for pricing. Do not invent a price.",
    "Happy to discuss pricing. The cost depends on the type of "
    "dispatching support and your operation.\n\n"
    "If you share a few details about your fleet, I can give you "
    "more relevant information."
),

make_outreach(
    "The prospect asks: 'Is there a setup fee?' Do not invent an answer.",
    "I don't want to give you inaccurate information without knowing "
    "which service setup you're considering.\n\n"
    "I can explain the applicable pricing and fees once I understand "
    "your requirements."
),

make_outreach(
    "The prospect says the price is higher than expected.",
    "I understand. The service needs to make sense financially for "
    "your operation.\n\n"
    "If you'd like, we can discuss your current setup and see whether "
    "there's a practical fit."
),

# Existing provider
make_outreach(
    "The prospect says: 'We already use a dispatch service.'",
    "Understood. If your current dispatching setup is working well, "
    "there's no need to change it.\n\n"
    "If you ever need additional support or another option in the "
    "future, feel free to reach out."
),

make_outreach(
    "The prospect says: 'We already have a dispatcher.'",
    "No problem. I appreciate you letting me know.\n\n"
    "If your situation changes or you ever need backup dispatching "
    "support, I'd be happy to connect."
),

make_outreach(
    "The prospect says they are satisfied with their current dispatcher.",
    "That's good to hear. If your current setup is working well, "
    "I completely understand staying with it.\n\n"
    "I'll leave the door open if you ever need another option."
),

# Not interested
make_outreach(
    "The prospect says: 'Not interested.'",
    "Understood. Thanks for getting back to me, {{name}}. "
    "I appreciate your time.\n\n"
    "Best wishes."
),

make_outreach(
    "The prospect says: 'Please don't contact me again.'",
    "Understood. I'll respect your request and won't follow up again.\n\n"
    "Thank you for letting me know."
),

make_outreach(
    "The prospect says they are not looking for dispatching right now.",
    "Understood. Thanks for letting me know.\n\n"
    "If your needs change in the future, you're welcome to reach out.\n\n"
    "Best,\n{{sender_name}}"
),

# Questions
make_outreach(
    "The prospect asks: 'What equipment do you work with?' Avoid unsupported claims.",
    "We can determine whether our service is suitable based on your "
    "equipment and operation.\n\n"
    "If you tell me what type of equipment you're running, I can "
    "discuss the fit with you."
),

make_outreach(
    "The prospect asks: 'Do you handle load searching?'",
    "Yes, load searching can be part of the dispatching support we "
    "provide. We can discuss your preferred equipment, lanes, and "
    "requirements to understand what you need."
),

make_outreach(
    "The prospect asks how dispatching support can help them.",
    "Dispatching support can help manage parts of the daily process "
    "such as finding suitable loads, coordinating load details, and "
    "keeping dispatch communication organized."
),

make_outreach(
    "The prospect asks what information you need from them.",
    "I'd need some basic information about your operation, such as "
    "your equipment type, fleet size, preferred lanes, and dispatching "
    "requirements. That helps us understand whether the service is a fit."
),

# Calls
make_outreach(
    "The prospect asks for a phone call.",
    "Sure. I'd be happy to speak with you.\n\n"
    "What time works best for you?"
),

make_outreach(
    "The prospect says: 'Call me this afternoon.'",
    "Absolutely. What time this afternoon works best for you?"
),

make_outreach(
    "The prospect wants to schedule a call next week.",
    "Sounds good. Let me know which day and time next week is convenient "
    "for you, and we can arrange the conversation."
),

# Busy
make_outreach(
    "The prospect says: 'I'm busy right now.'",
    "No problem. I understand.\n\n"
    "If there's a better time to connect, just let me know."
),

make_outreach(
    "The prospect says: 'I don't have time to talk.'",
    "Understood. I'll keep it brief and won't take more of your time.\n\n"
    "If dispatching support becomes relevant later, feel free to reach out."
),

# Information
make_outreach(
    "The prospect asks you to send information by email.",
    "Of course. I'll keep the information concise and focused on our "
    "truck dispatching service.\n\n"
    "If you share your fleet size and equipment type, I can make it "
    "more relevant to your operation."
),

make_outreach(
    "The prospect says: 'Just send me your company information.'",
    "Absolutely. I'll send over a concise overview of our truck "
    "dispatching service and the type of support we provide."
),

# Objections
make_outreach(
    "The prospect says: 'We do our own dispatching.'",
    "Understood. If handling dispatching internally is working well "
    "for your operation, that makes sense.\n\n"
    "If you ever want to compare your current process with outside "
    "support, I'd be happy to connect."
),

make_outreach(
    "The prospect says: 'I don't trust dispatching companies.'",
    "I understand the concern. Choosing a dispatching service is "
    "something you should evaluate carefully.\n\n"
    "I'm happy to answer your questions about how our service works "
    "so you can decide whether it's relevant."
),

make_outreach(
    "The prospect says: 'We've had a bad experience with dispatchers.'",
    "I understand why that would make you cautious.\n\n"
    "If you're open to it, I can explain how our dispatching process "
    "works and answer any questions you have."
),

make_outreach(
    "The prospect says: 'I don't want to switch providers.'",
    "Understood. If you're happy with your current provider, there's "
    "no pressure to change.\n\n"
    "I'll be happy to connect if your needs change in the future."
),

# Tone rewriting
make_outreach(
    "Rewrite this aggressive sentence to sound professional: "
    "'You need our dispatching service if you want to make more money.'",
    "You may find our dispatching support useful if you're looking "
    "for assistance with your current operation."
),

make_outreach(
    "Rewrite this email to sound less sales-focused.",
    "Hi {{name}},\n\n"
    "I wanted to reach out and learn how you're currently handling "
    "dispatching at {{company}}.\n\n"
    "We provide dispatching support for carriers, and I thought it "
    "would be useful to introduce ourselves.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Shorten a long truck dispatching outreach email.",
    "Subject: Quick question\n\n"
    "Hi {{name}},\n\n"
    "How are you currently handling dispatching at {{company}}?\n\n"
    "We provide truck dispatching support and would be happy to "
    "connect if you're exploring options.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Rewrite a generic outreach email to sound more conversational.",
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to reach out. I'm curious "
    "how you're currently handling dispatching for your fleet.\n\n"
    "We provide dispatching support for carriers and I'd be happy "
    "to compare notes if useful.\n\n"
    "Best,\n{{sender_name}}"
),

# Feedback learning
make_outreach(
    "Feedback: The email is too long. Make it shorter.",
    "Hi {{name}},\n\n"
    "I wanted to reach out regarding dispatching support for {{company}}.\n\n"
    "How are you currently handling dispatching for your fleet?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Feedback: The email sounds robotic. Make it natural.",
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to introduce myself. "
    "I'm curious how you're currently handling dispatching.\n\n"
    "We provide dispatching support for carriers if you ever want "
    "to explore another option.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Feedback: Remove the exaggerated claim that we are the best.",
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service. We support "
    "carriers with their day-to-day dispatching and load coordination.\n\n"
    "Would you be open to a quick conversation?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Feedback: The call to action is too aggressive. Make it softer.",
    "Hi {{name}},\n\n"
    "I came across {{company}} and wanted to introduce our dispatching "
    "service.\n\n"
    "If this is relevant to your operation, I'd be happy to have a "
    "quick conversation.\n\n"
    "Best,\n{{sender_name}}"
),

# Conversation handling
make_outreach(
    "The prospect asks: 'Where did you find my company?'",
    "I came across {{company}} while researching trucking companies "
    "and wanted to introduce our dispatching service."
),

make_outreach(
    "The prospect asks: 'Why are you contacting me?'",
    "I reached out because {{company}} is a trucking operation and "
    "we provide dispatching support for carriers.\n\n"
    "I wanted to see how you're currently handling dispatching and "
    "whether an outside option might be relevant."
),

make_outreach(
    "The prospect asks: 'Are you a broker?'",
    "We're offering truck dispatching support rather than approaching "
    "you as a freight broker. I'd be happy to explain our service "
    "and how the dispatching relationship works."
),

make_outreach(
    "The prospect says: 'I'm interested, but I need to think about it.'",
    "Of course. Take your time.\n\n"
    "If you have any questions while considering it, feel free to "
    "send them over and I'll be happy to answer."
),

make_outreach(
    "The prospect says: 'Maybe later.'",
    "Understood. Timing matters.\n\n"
    "I'll leave the door open, and if dispatching support becomes "
    "relevant later, you're welcome to reach out."
),

make_outreach(
    "The prospect asks for references or proof of service.",
    "I understand. It's reasonable to evaluate a service before "
    "moving forward.\n\n"
    "I can provide information that we actually have available "
    "and answer any specific questions about our process."
),

make_outreach(
    "The prospect asks for guaranteed results. Do not make guarantees.",
    "I wouldn't want to promise a guaranteed result because every "
    "operation is different.\n\n"
    "What I can do is explain our dispatching process and what type "
    "of support we provide so you can evaluate whether it fits."
),

make_outreach(
    "The prospect gives a positive response but does not ask a question.",
    "Thanks, {{name}}. I appreciate the response.\n\n"
    "If you're open to it, we can have a quick conversation about "
    "your current dispatching setup and what you're looking for."
),

make_outreach(
    "The prospect says they want to start immediately.",
    "Great. Before getting started, I'd like to understand your fleet, "
    "equipment, lanes, and dispatching requirements so we can confirm "
    "the setup and next steps."
),

make_outreach(
    "The prospect asks what the next step is.",
    "The next step would be to understand your fleet and dispatching "
    "requirements, then discuss whether our service is a suitable fit."
),

# Email quality
make_outreach(
    "Write an outreach email with no exaggerated claims.",
    "Subject: Dispatching support\n\n"
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service. We help "
    "carriers with load searching and dispatch coordination.\n\n"
    "If you're open to discussing your current setup, I'd be happy "
    "to connect.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a cold email using a question as the opening.",
    "Subject: Quick question\n\n"
    "Hi {{name}},\n\n"
    "How are you currently handling dispatching for {{company}}?\n\n"
    "We provide truck dispatching support for carriers, and I'd be "
    "happy to explain our service if you're exploring options.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a professional email that avoids excessive marketing language.",
    "Hi {{name}},\n\n"
    "I wanted to introduce our truck dispatching service and learn "
    "a little about your current dispatching process.\n\n"
    "If outside support is something you're considering, I'd be happy "
    "to have a conversation.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a short outreach email with one clear call to action.",
    "Subject: Dispatching question\n\n"
    "Hi {{name}},\n\n"
    "We provide truck dispatching support for carriers. "
    "How are you currently handling dispatching at {{company}}?\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write an email that respects a prospect who is already satisfied.",
    "Hi {{name}},\n\n"
    "I understand if you're happy with your current dispatching setup. "
    "I just wanted to introduce our service in case you ever need "
    "another option.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a polite response when the prospect declines.",
    "Thanks for letting me know, {{name}}. I appreciate the response "
    "and respect your decision.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a short response when the prospect asks you to stop following up.",
    "Understood. I respect your request and won't follow up again.\n\n"
    "Thank you for letting me know."
),

make_outreach(
    "Write a follow-up that adds no new pressure.",
    "Hi {{name}},\n\n"
    "Just checking back on my earlier message. If dispatching support "
    "isn't relevant right now, completely understood.\n\n"
    "Best,\n{{sender_name}}"
),

make_outreach(
    "Write a response to a prospect who asks for a few days to decide.",
    "Of course. Take the time you need.\n\n"
    "If any questions come up while you're deciding, feel free to "
    "send them over."
),

make_outreach(
    "Write a professional response to a prospect who asks a question "
    "you cannot answer without more information.",
    "I'd be happy to give you an accurate answer. I just need a little "
    "more information about your fleet and current setup first."
),
]

# ================================================================
# PART 3 — Operational & specialized correspondence (expansion)
# ================================================================

more_examples = []

# ----------------------------------------------------------------
# 1. Rate confirmation & load tendering
# ----------------------------------------------------------------

more_examples += [
    make_general(
        "Write an email confirming a rate for a load and requesting "
        "signed confirmation.",
        "Subject: Rate Confirmation — Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "Please find the rate confirmation for load {{load_id}} "
        "attached, covering pickup at {{origin}} and delivery to "
        "{{destination}} at the agreed rate of {{rate}}.\n\n"
        "Kindly review, sign, and return at your earliest convenience "
        "so we can proceed with scheduling.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email tendering a new load to a carrier and asking "
        "if they can cover it.",
        "Subject: Load Available — {{origin}} to {{destination}}\n\n"
        "Dear {{name}},\n\n"
        "We have a load available from {{origin}} to {{destination}}, "
        "picking up on {{pickup_date}}. The proposed rate is "
        "{{rate}}.\n\n"
        "Please let me know if this works with your current schedule "
        "and equipment.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "The carrier asks for a higher rate than originally offered. "
        "Write a professional response.",
        "Subject: Re: Rate for Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "Thank you for the update. Let me check with the shipper on "
        "the possibility of adjusting the rate for load {{load_id}} "
        "and follow up with you shortly.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email confirming pickup and delivery appointment "
        "times for a scheduled load.",
        "Subject: Appointment Times — Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "This is to confirm the appointment times for load "
        "{{load_id}}: pickup at {{origin}} on {{pickup_date}} at "
        "{{pickup_time}}, and delivery at {{destination}} on "
        "{{delivery_date}} at {{delivery_time}}.\n\n"
        "Please let me know if any of these times need to be "
        "adjusted.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a short email asking a carrier to send updated "
        "insurance and authority documents before dispatching a load.",
        "Subject: Documentation Needed Before Dispatch\n\n"
        "Dear {{name}},\n\n"
        "Before we can dispatch load {{load_id}}, could you please "
        "send over your current certificate of insurance and operating "
        "authority documentation?\n\n"
        "Once received, we can move forward with scheduling.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email letting a carrier know a load has been "
        "cancelled by the shipper.",
        "Subject: Load {{load_id}} Cancelled\n\n"
        "Dear {{name}},\n\n"
        "I want to let you know that load {{load_id}}, originally "
        "scheduled for pickup on {{pickup_date}}, has been cancelled "
        "by the shipper. I apologize for any inconvenience this may "
        "cause and will reach out as soon as another suitable load is "
        "available.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a professional email offering a backhaul load to a "
        "carrier who just delivered nearby.",
        "Subject: Backhaul Opportunity Near {{destination}}\n\n"
        "Dear {{name}},\n\n"
        "Since your driver is delivering near {{destination}}, I "
        "wanted to check if you would be interested in a backhaul load "
        "from that area at {{rate}}.\n\n"
        "Let me know if this fits your route.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
]

# ----------------------------------------------------------------
# 2. Billing & invoicing correspondence
# ----------------------------------------------------------------

more_examples += [
    make_general(
        "Write a professional email sending an invoice for a completed "
        "load.",
        "Subject: Invoice for Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "Please find attached the invoice for load {{load_id}}, "
        "delivered on {{delivery_date}}. The total amount due is "
        "{{amount}}, payable within {{payment_terms}}.\n\n"
        "Please let me know if you require any additional "
        "documentation for processing.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a polite email following up on an overdue invoice "
        "without sounding aggressive.",
        "Subject: Follow-Up — Invoice {{invoice_id}}\n\n"
        "Dear {{name}},\n\n"
        "I wanted to follow up regarding invoice {{invoice_id}}, which "
        "appears to still be outstanding. If payment has already been "
        "sent, please disregard this message.\n\n"
        "Otherwise, could you let me know an expected payment date?\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email explaining a discrepancy between the quoted "
        "rate and the invoiced amount.",
        "Subject: Invoice Discrepancy — Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "I noticed a difference between the originally quoted rate and "
        "the amount on invoice {{invoice_id}} for load {{load_id}}. "
        "Could you help clarify the reason for this adjustment?\n\n"
        "I appreciate your help in resolving this quickly.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email confirming that a payment has been received.",
        "Subject: Payment Received — Invoice {{invoice_id}}\n\n"
        "Dear {{name}},\n\n"
        "This is to confirm that we have received payment for invoice "
        "{{invoice_id}}. Thank you for your promptness.\n\n"
        "Please let us know if you need a receipt for your records.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email to a factoring company requesting confirmation "
        "of an assigned invoice.",
        "Subject: Invoice Assignment Confirmation — {{invoice_id}}\n\n"
        "Dear {{name}},\n\n"
        "Could you please confirm that invoice {{invoice_id}}, assigned "
        "under our factoring agreement, has been received and is being "
        "processed?\n\n"
        "Thank you for your assistance.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a professional email notifying a client of an upcoming "
        "rate increase due to fuel costs.",
        "Subject: Notice of Fuel Surcharge Adjustment\n\n"
        "Dear {{name}},\n\n"
        "Due to recent increases in fuel costs, we will be applying an "
        "updated fuel surcharge to shipments beginning "
        "{{effective_date}}.\n\n"
        "We appreciate your understanding and are happy to discuss "
        "this further if you have questions.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
]

# ----------------------------------------------------------------
# 3. Compliance & safety documentation
# ----------------------------------------------------------------

more_examples += [
    make_general(
        "Write an email requesting a carrier's updated safety rating "
        "and DOT number for onboarding.",
        "Subject: Onboarding Documentation Request\n\n"
        "Dear {{name}},\n\n"
        "To complete your onboarding with us, could you please provide "
        "your current DOT number, MC number, and most recent safety "
        "rating?\n\n"
        "Once received, we can finalize your setup in our system.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a reminder email about an expiring insurance "
        "certificate.",
        "Subject: Insurance Certificate Expiring Soon\n\n"
        "Dear {{name}},\n\n"
        "Our records show that your certificate of insurance is set to "
        "expire on {{expiry_date}}. To avoid any interruption in "
        "dispatching, please send an updated certificate before that "
        "date.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email confirming receipt of a signed carrier "
        "agreement.",
        "Subject: Carrier Agreement Received\n\n"
        "Dear {{name}},\n\n"
        "Thank you for returning the signed carrier agreement. We have "
        "it on file and will proceed with setting up your account.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email asking a carrier to correct a discrepancy "
        "found during a routine compliance review.",
        "Subject: Compliance Review — Action Needed\n\n"
        "Dear {{name}},\n\n"
        "During a routine review of your file, we noticed that your "
        "{{document_type}} appears to be outdated. Could you please "
        "send an updated copy at your earliest convenience?\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a professional email reminding drivers about a required "
        "hours-of-service policy update.",
        "Subject: Hours-of-Service Policy Update\n\n"
        "Dear {{name}},\n\n"
        "This is a reminder that our hours-of-service reporting policy "
        "has been updated effective {{effective_date}}. Please review "
        "the attached summary and reach out with any questions.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
]

# ----------------------------------------------------------------
# 4. Driver-facing dispatch communication
# ----------------------------------------------------------------

more_examples += [
    make_general(
        "Write a short, clear message to a driver with pickup details "
        "for a new load.",
        "Load {{load_id}} — Pickup Details\n\n"
        "Hi {{driver_name}},\n\n"
        "You're scheduled to pick up load {{load_id}} at {{origin}} on "
        "{{pickup_date}} at {{pickup_time}}. Delivery is to "
        "{{destination}} by {{delivery_date}}.\n\n"
        "Please confirm once you're on your way.\n\n"
        "Thanks,\n{{sender_name}}"
    ),
    make_general(
        "Write a message to a driver asking for a status update on a "
        "load that's due for delivery soon.",
        "Hi {{driver_name}},\n\n"
        "Just checking in on load {{load_id}} — could you let me know "
        "your current location and estimated arrival time at "
        "{{destination}}?\n\n"
        "Thanks for the update.\n\n"
        "{{sender_name}}"
    ),
    make_general(
        "Write a message to a driver acknowledging a reported delay "
        "and letting them know the customer has been notified.",
        "Hi {{driver_name}},\n\n"
        "Thank you for letting me know about the delay on load "
        "{{load_id}}. I've informed the customer of the updated "
        "delivery time, so there's nothing further you need to do on "
        "that front.\n\n"
        "Drive safe.\n\n"
        "{{sender_name}}"
    ),
    make_general(
        "Write a message thanking a driver for handling a difficult "
        "delivery professionally.",
        "Hi {{driver_name}},\n\n"
        "I wanted to thank you for how you handled the delivery at "
        "{{destination}} today. Your professionalism made a real "
        "difference, and I appreciate it.\n\n"
        "{{sender_name}}"
    ),
    make_general(
        "Write a message to a driver with new dispatch instructions "
        "after a load has been rerouted.",
        "Hi {{driver_name}},\n\n"
        "Load {{load_id}} has been rerouted. Please deliver to "
        "{{new_destination}} instead of the original destination, with "
        "the same delivery window.\n\n"
        "Let me know if you have any questions.\n\n"
        "{{sender_name}}"
    ),
]

# ----------------------------------------------------------------
# 5. Escalation & service recovery
# ----------------------------------------------------------------

more_examples += [
    make_general(
        "Write a professional email escalating an unresolved issue to "
        "a manager after multiple follow-ups.",
        "Subject: Escalation — Unresolved Issue on Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "I wanted to escalate the ongoing issue regarding load "
        "{{load_id}}, as it remains unresolved after several "
        "follow-ups. Could we schedule a brief call to work through a "
        "resolution?\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email apologizing for a missed pickup and outlining "
        "the corrective steps being taken.",
        "Subject: Apology — Missed Pickup on Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "I want to sincerely apologize for the missed pickup on load "
        "{{load_id}}. We are reviewing what went wrong and putting "
        "{{corrective_action}} in place to prevent it from happening "
        "again.\n\n"
        "Please let me know how I can help make this right.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email responding calmly to a client who is upset "
        "about a late delivery.",
        "Subject: Re: Late Delivery — Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "I understand your frustration regarding the delayed delivery "
        "of load {{load_id}}, and I apologize for the inconvenience "
        "this has caused. Here is what happened and what we are doing "
        "to address it: {{corrective_action}}.\n\n"
        "Please let me know if there's anything further I can do.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email to a client explaining that an issue has been "
        "fully resolved and offering a gesture of goodwill.",
        "Subject: Resolution Update — Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "I'm glad to let you know that the issue with load {{load_id}} "
        "has been fully resolved. As a gesture of goodwill, we would "
        "like to offer {{goodwill_offer}} on your next shipment.\n\n"
        "Thank you for your patience throughout this process.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
]

# ----------------------------------------------------------------
# 6. Backhaul / lane-specific outreach
# ----------------------------------------------------------------

more_examples += [
    make_outreach(
        "Write a short outreach email to a carrier about a specific "
        "lane you run frequently.",
        "Subject: Freight on the {{origin}} to {{destination}} Lane\n\n"
        "Hi {{name}},\n\n"
        "We consistently move freight on the {{origin}} to "
        "{{destination}} lane and wanted to see if that fits your "
        "routes.\n\n"
        "Would you be open to a quick conversation about steady lane "
        "opportunities?\n\n"
        "Best,\n{{sender_name}}"
    ),
    make_outreach(
        "Write an outreach email to a carrier who specializes in "
        "reefer freight.",
        "Subject: Reefer Freight Opportunities\n\n"
        "Hi {{name}},\n\n"
        "I noticed {{company}} runs reefer equipment, and we regularly "
        "have temperature-controlled loads available.\n\n"
        "Would it make sense to connect and see if there's a fit for "
        "your fleet?\n\n"
        "Best,\n{{sender_name}}"
    ),
    make_outreach(
        "Write a brief outreach message for a carrier with flatbed "
        "capacity.",
        "Subject: Flatbed Capacity Needed\n\n"
        "Hi {{name}},\n\n"
        "We're working with shippers who regularly need flatbed "
        "capacity and wanted to check if {{company}} has availability "
        "in the coming weeks.\n\n"
        "Let me know if you'd like to discuss.\n\n"
        "Best,\n{{sender_name}}"
    ),
    make_outreach(
        "Write an outreach email re-engaging a carrier who hasn't "
        "hauled a load in a few months.",
        "Subject: Checking Back In\n\n"
        "Hi {{name}},\n\n"
        "It's been a little while since {{company}} last hauled for "
        "us, and I wanted to check in and see how things have been "
        "going.\n\n"
        "We'd love to work together again if the timing is right.\n\n"
        "Best,\n{{sender_name}}"
    ),
]

# ----------------------------------------------------------------
# 7. Renewals, holidays & schedule notices
# ----------------------------------------------------------------

more_examples += [
    make_general(
        "Write a professional email letting clients know about "
        "holiday office hours.",
        "Subject: Holiday Schedule Notice\n\n"
        "Dear {{name}},\n\n"
        "Please note that our office will observe modified hours on "
        "{{holiday_dates}}. Dispatch support will remain available for "
        "urgent matters during this time.\n\n"
        "Wishing you a safe and restful holiday.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a professional email about a weather-related delay "
        "affecting scheduled loads.",
        "Subject: Weather Delay Notice — Load {{load_id}}\n\n"
        "Dear {{name}},\n\n"
        "Due to severe weather conditions along the route, load "
        "{{load_id}} may experience a delay. We are monitoring the "
        "situation closely and will keep you updated.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write an email proposing renewal of a carrier partnership "
        "agreement before it expires.",
        "Subject: Partnership Renewal — {{company}}\n\n"
        "Dear {{name}},\n\n"
        "Our current agreement is set to expire on {{expiry_date}}, "
        "and we would like to discuss renewing our partnership for the "
        "coming year.\n\n"
        "Would you be available for a brief call to go over the "
        "details?\n\n"
        "Best regards,\n{{sender_name}}"
    ),
    make_general(
        "Write a short email requesting a testimonial from a "
        "long-term, satisfied client.",
        "Subject: Would You Share Your Experience?\n\n"
        "Dear {{name}},\n\n"
        "We've enjoyed working with {{company}} and would be grateful "
        "if you'd be willing to share a short testimonial about your "
        "experience with our dispatching service.\n\n"
        "No pressure at all, and thank you either way.\n\n"
        "Best regards,\n{{sender_name}}"
    ),
]

# ----------------------------------------------------------------
# 8. Short-form replies (SMS / chat style)
# ----------------------------------------------------------------

more_examples += [
    make_general(
        "Write a very short text-style reply confirming a load has "
        "been booked.",
        "Hi {{name}}, load {{load_id}} is confirmed and booked. "
        "I'll send full details shortly. Thanks!"
    ),
    make_general(
        "Write a brief text-style message letting a client know a "
        "driver is running about an hour behind.",
        "Hi {{name}}, quick update — the driver on load {{load_id}} is "
        "running about an hour behind schedule. Will keep you posted."
    ),
    make_general(
        "Write a short text-style reply asking a carrier to confirm "
        "they received the rate confirmation.",
        "Hi {{name}}, just checking you received the rate confirmation "
        "for load {{load_id}}. Let me know if you have any questions."
    ),
    make_general(
        "Write a brief, polite text-style reply declining a rate "
        "request that's too low.",
        "Hi {{name}}, thanks for the offer, but {{rate}} is below what "
        "we can accept on this lane right now. Happy to revisit if "
        "anything changes."
    ),
]

# ================================================================
# Combine, de-duplicate, validate
# ================================================================

all_examples = examples + new_examples + more_examples

unique_examples = []
seen = set()
for item in all_examples:
    key = json.dumps(item, sort_keys=True)
    if key not in seen:
        seen.add(key)
        unique_examples.append(item)

if len(unique_examples) < MIN_EXAMPLES:
    raise RuntimeError(
        f"Only {len(unique_examples)} unique examples were generated; "
        f"at least {MIN_EXAMPLES} are required before training."
    )

# Keep the dataset size fixed and reproducible.
unique_examples = unique_examples[:MIN_EXAMPLES]

random.seed(RANDOM_SEED)
random.shuffle(unique_examples)

n_val = round(len(unique_examples) * VALIDATION_SPLIT)
validation_examples = unique_examples[:n_val]
train_examples = unique_examples[n_val:]

# ================================================================
# Write dataset files
# ================================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)

with TRAIN_FILE.open("w", encoding="utf-8") as f:
    for item in train_examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with VALIDATION_FILE.open("w", encoding="utf-8") as f:
    for item in validation_examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Dataset build complete.")
print(f"  Unique examples:    {len(unique_examples)}")
print(f"  Training examples:  {len(train_examples)} -> {TRAIN_FILE}")
print(f"  Validation examples:{len(validation_examples):>4} -> {VALIDATION_FILE}")
