"""
build_dataset_v3.py

Generates a clean, fully professional dataset of business emails and
responses for a truck dispatching service. Replaces the v2 dataset.

Every example is written in polished, formal-professional business
English: correct grammar, correct punctuation, consistent formatting,
no slang, no unsupported claims, no aggressive sales language.

Output:
    data/train.jsonl
    data/validation.jsonl
"""

import json
import random
from pathlib import Path

random.seed(7)

DATA_DIR = Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"

SYSTEM_PROMPT = (
    "You are a professional business correspondence assistant for a "
    "truck dispatching service. Write clear, courteous, and fully "
    "professional emails and responses. Use correct grammar and "
    "punctuation at all times. Do not make unsupported claims. Do "
    "not use aggressive sales language, slang, or casual phrasing."
)


def make(user, assistant):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


examples = []

# ============================================================
# 1. FIRST CONTACT / INTRODUCTION
# ============================================================

examples += [
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
        "Write a brief, professional second follow-up after no reply.",
        "Subject: Checking In\n\n"
        "Dear {{name}},\n\n"
        "I am following up once more regarding dispatching support "
        "for {{company}}. Please let me know if this would be a "
        "relevant conversation to have at this time.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
        "The prospect says they are not interested at this time. "
        "Write a professional and courteous response.",
        "Dear {{name}},\n\n"
        "Thank you for letting me know, and I appreciate your "
        "response. I understand this may not be the right time.\n\n"
        "Should your situation change, I would welcome the "
        "opportunity to reconnect.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make(
        "The prospect asks to be removed from future outreach. Write "
        "a professional response confirming the request.",
        "Dear {{name}},\n\n"
        "Thank you for letting me know. I have noted your request and "
        "will remove {{company}} from future outreach.\n\n"
        "I wish you continued success with your operations.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make(
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
    make(
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
    make(
        "Write a confirmation email for a scheduled call.",
        "Subject: Confirming Our Call — {{proposed_time}}\n\n"
        "Dear {{name}},\n\n"
        "I am writing to confirm our call scheduled for "
        "{{proposed_time}}. Please let me know if you need to "
        "reschedule, and I will be glad to accommodate.\n\n"
        "I look forward to speaking with you.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make(
        "Write a professional email requesting to reschedule a call.",
        "Subject: Request to Reschedule\n\n"
        "Dear {{name}},\n\n"
        "I apologize for the inconvenience, but I need to reschedule "
        "our upcoming call. Could you let me know a few times that "
        "would work well for you this week?\n\n"
        "Thank you for your understanding.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
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
    make(
        "Write a professional check-in email for a long-standing "
        "client, without a specific sales purpose.",
        "Subject: Checking In\n\n"
        "Dear {{name}},\n\n"
        "I wanted to check in and see how things have been going with "
        "{{company}}'s operations. Please let me know if there is "
        "anything we can assist with or improve upon on our end.\n\n"
        "Best regards,\n{{sender_name}}",
    ),
    make(
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

random.shuffle(examples)

n_total = len(examples)
n_val = max(1, round(n_total * 0.15))

validation = examples[:n_val]
train = examples[n_val:]

DATA_DIR.mkdir(parents=True, exist_ok=True)

with TRAIN_FILE.open("w", encoding="utf-8") as f:
    for ex in train:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with VALIDATION_FILE.open("w", encoding="utf-8") as f:
    for ex in validation:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Total examples: {n_total}")
print(f"Train: {len(train)} -> {TRAIN_FILE}")
print(f"Validation: {len(validation)} -> {VALIDATION_FILE}")
