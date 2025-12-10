import csv
import json
import logging
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Literal

from bookgap.util import setup_logging

LOGGER = logging.getLogger(__name__)

SURVEY_QUESTIONS = [
    "interesting?",
    "well-written?",
    "worth publishing?"
]

# BOOK_QUESTION_LABELS = {
#     "1": {"Display": "Not at all"},
#     "2": {"Display": "Slightly"},
#     "3": {"Display": "Moderately"},
#     "4": {"Display": "Very"},
#     "5": {"Display": "Extremely"},
# }

BOOK_QUESTION_LABELS = {
    "1": {"Display": "Not at all"},
    "2": {"Display": "Slightly"},
    "3": {"Display": "Somewhat"},
    "4": {"Display": "Moderately"},
    "5": {"Display": "Quite"},
    "6": {"Display": "Very"},
    "7": {"Display": "Extremely"},
}

# Toggle the per-book question style between Qualtrics sliders and a Likert matrix
QUESTION_STYLE: Literal["radio", "slider"] = "radio"

SLIDER_PROMPT = "Do you personally think this book is..."

def generate_qualtrics_id(prefix, length=15):
    chars = string.ascii_letters + string.digits
    return prefix + ''.join(random.choice(chars) for _ in range(length))

def format_qualtrics_txt(books: List[Dict[str, str]]) -> str:
    """Generate Qualtrics TXT content."""
    lines = ["[[AdvancedFormat]]"]
    
    # Single block for all questions to allow randomization
    lines.append("[[Block:Content]]")
    
    for i, book in enumerate(books):
        # Per-book rating question (slider or matrix) with Title, Abstract, and three statements (1-7 scale)
        question_tag = "Matrix" if QUESTION_STYLE == "radio" else "Slider"
        lines.append(f"[[Question:{question_tag}]]")
        
        # Combine Title, Abstract, and Prompt into the question text
        lines.append(f"<b>Title:</b> {book['title']}<br><br>")
        # Replace literal \n with <br> for HTML display
        abstract_html = book['abstract'].replace('\\n', '<br>')
        lines.append(f"<b>Abstract:</b> {abstract_html}")
        lines.append("<br><br>")
        
        if SLIDER_PROMPT:
            # Render the prompt in bold so it stands out with the Title/Abstract
            lines.append(f"<b>{SLIDER_PROMPT}</b>")
            
        # Define the three statements
        lines.append("[[Choices]]")
        for question in SURVEY_QUESTIONS:
            lines.append(question)
        # Define the 1-7 scale answers
        lines.append("[[Answers]]")
        for j in range(1, 8):
            lines.append(str(j))
        lines.append("")
        
        # Add Page Break after each book
        if i < len(books) - 1:
            lines.append("[[PageBreak]]")
        
    return "\n".join(lines)


def build_book_question_payload(book: Dict[str, str], qid_sl: str, survey_id: str) -> Dict[str, Any]:
    """Return the QSF question payload for a single book based on QUESTION_STYLE."""
    abstract_html = book['abstract'].replace('\\n', '<br>')
    q_text = f"<b>Title:</b> {book['title']}<br><br>\n<b>Abstract:</b> {abstract_html}\n\n<br><br>"
    if SLIDER_PROMPT:
        q_text += f"<b>{SLIDER_PROMPT}</b>"

    desc_text = f"Title: {book['title']} Abstract: {book['abstract']}"
    payload_base = {
        "QuestionText": q_text,
        "QuestionID": qid_sl,
        "QuestionDescription": desc_text[:100],
        "DataVisibility": {"Private": False, "Hidden": False},
    }

    if QUESTION_STYLE == "slider":
        return {
            "SurveyID": survey_id,
            "Element": "SQ",
            "PrimaryAttribute": qid_sl,
            "SecondaryAttribute": desc_text[:100],
            "TertiaryAttribute": None,
            "Payload": {
                **payload_base,
                "QuestionType": "Slider",
                "Selector": "HSLIDER",
                "Configuration": {
                    "QuestionDescriptionOption": "UseText",
                    "CSSliderMin": 1,
                    "CSSliderMax": 7,
                    "GridLines": 6,
                    "SnapToGrid": True,
                    "NumDecimals": "0",
                    "CustomStart": False,
                    "SliderStartPositions": {
                        "1": 0.5,
                        "2": 0.5,
                        "3": 0.5
                    },
                    "NotApplicable": False,
                    "MobileFirst": True
                },
                "Choices": {
                    str(i): {"Display": q} for i, q in enumerate(SURVEY_QUESTIONS, start=1)
                },
                "ChoiceOrder": [i for i in range(1, len(SURVEY_QUESTIONS) + 1)],
                "Validation": {"Settings": {"Type": "None"}},
                "Language": [],
                "NextChoiceId": len(SURVEY_QUESTIONS) + 1,
                "NextAnswerId": 1,
                "Labels": BOOK_QUESTION_LABELS,
                "Answers": {
                    "1": {"Display": 1},
                    "2": {"Display": 2},
                    "3": {"Display": 3},
                    "4": {"Display": 4},
                    "5": {"Display": 5},
                    "6": {"Display": 6},
                    "7": {"Display": 7}
                },
                "DataExportTag": qid_sl,
            }
        }

    if QUESTION_STYLE == "radio":
        return {
            "SurveyID": survey_id,
            "Element": "SQ",
            "PrimaryAttribute": qid_sl,
            "SecondaryAttribute": desc_text[:100],
            "TertiaryAttribute": None,
            "Payload": {
                **payload_base,
                "QuestionType": "Matrix",
                "Selector": "Likert",
                "SubSelector": "SingleAnswer",
                "Configuration": {
                    "QuestionDescriptionOption": "UseText",
                    "TextPosition": "inline",
                    "ChoiceColumnWidth": 25,
                    "RepeatHeaders": "none",
                    "WhiteSpace": "OFF",
                    "MobileFirst": True,
                },
                "Choices": {
                    str(i): {"Display": q} for i, q in enumerate(SURVEY_QUESTIONS, start=1)
                },
                "ChoiceOrder": [str(i) for i in range(1, len(SURVEY_QUESTIONS) + 1)],
                "Answers": BOOK_QUESTION_LABELS,
                "AnswerOrder": list(range(1, 8)),
                "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
                "Language": [],
                "NextChoiceId": len(SURVEY_QUESTIONS) + 1,
                "NextAnswerId": 8,
                "ChoiceDataExportTags": False,
                "DataExportTag": qid_sl,
            }
        }

    raise ValueError(f"Unsupported QUESTION_STYLE: {QUESTION_STYLE}")

def build_qsf(books: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build a Qualtrics QSF JSON payload mirroring our survey blocks.
    Each book becomes a block with a descriptive text and a per-book rating question
    whose style is controlled by QUESTION_STYLE.
    """
    SURVEY_DESCRIPTION = "Evaluate AI-generated and human-authored textbook proposals."

    # Minimal QSF scaffold
    survey_id = generate_qualtrics_id("SV_")
    blocks = []
    questions = []
    flow = []
    block_id_prefix = 1000
    qid_counter = 1

    def new_qid():
        nonlocal qid_counter
        q = f"QID{qid_counter}"
        qid_counter += 1
        return q

    # Single content block for all books
    content_block_id = "BL_Content"
    content_block_elements = []

    # Create questions for each book
    for idx, book in enumerate(books, start=1):
        qid_sl = new_qid()
        questions.append(build_book_question_payload(book, qid_sl, survey_id))
        content_block_elements.append({"Type": "Question", "QuestionID": qid_sl})
        
        # # Add a Page Break after each book (except the last one, optionally)
        # if idx < len(books):
        #      content_block_elements.append({"Type": "Page Break"})

    # Create the single content block
    blocks.append({
        "Type": "Standard",
        "SubType": "",
        "Description": "Book Ratings",
        "ID": content_block_id,
        "BlockElements": content_block_elements,
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "RandomWithXPerPage",
            "Randomization": {
                "Advanced": {
                    "QuestionsPerPage": 1
                }
            }
        }
    })

    # Create Welcome block
    welcome_block_id = "BL_Welcome"
    welcome_block_elements = []
    
    # Create welcome / descriptive text
    qid_welcome = new_qid()
    welcome_html = (
        "<span style=\"font-size:19px;\">Survey: Textbook Proposal Evaluation</span><br>\n"
        "<br>\n"
        "<span style=\"font-size:16px;\"><strong>Goal:</strong> This survey aims to evaluate the performance of an end-to-end system for proposing new textbooks, against human-authored proposals.<br>\n"
        "<br>\n"
        "<strong>What to expect:</strong> You will be presented with a randomized list of proposals (titles and abstracts). Some are generated by our automated system and others are written by humans. You will be asked to evaluate each proposal based on:</span><br><br>\n"
        "<table align=\"left\" border=\"0\" cellpadding=\"0\" cellspacing=\"10\">\n\t<tbody>\n\t\t<tr>\n\t\t\t<td><span style=\"font-size:16px;\"><em>Topic Selection</em></span></td>\n\t\t\t<td><span style=\"font-size:16px;\">Is the book interesting to you?</span></td>\n\t\t</tr>\n\t\t<tr>\n\t\t\t<td><span style=\"font-size:16px;\"><em>Writing Quality</em></span></td>\n\t\t\t<td><span style=\"font-size:16px;\">Is the proposal is well-written?</span></td>\n\t\t</tr>\n\t\t<tr>\n\t\t\t<td><span style=\"font-size:16px;\"><em>Publishing Value</em></span></td>\n\t\t\t<td><span style=\"font-size:16px;\">Is it worth writing and publishing the corresponding book?</span></td>\n\t\t</tr>\n\t</tbody>\n</table>\n<br type=\"_moz\">\n<br>\n<br>\n<br>\n<br>\n<span style=\"font-size:16px;\"><strong>Estimated time:</strong> 20 minutes<br>\n<br>\nThank you for participating in this study!</span>"
    )
    questions.append({
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": qid_welcome,
        "SecondaryAttribute": "Survey: Textbook Proposal Evaluation",
        "TertiaryAttribute": None,
        "Payload": {
            "QuestionText": welcome_html,
            "DefaultChoices": False,
            "DataExportTag": qid_welcome.replace('QID', 'Q'),
            "QuestionID": qid_welcome,
            "QuestionType": "DB",
            "Selector": "TB",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": "Survey: Textbook Proposal Evaluation",
            "ChoiceOrder": [],
            "Validation": {"Settings": {"Type": "None"}},
        }
    })
    welcome_block_elements.append({"Type": "Question", "QuestionID": qid_welcome})
    
    # Create "Contact email" text entry question
    qid_email = new_qid()
    questions.append({
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": qid_email,
        "SecondaryAttribute": "Contact email",
        "TertiaryAttribute": None,
        "Payload": {
            "QuestionText": "Contact email (optional, if you would like us to send the results of your survey)",
            "QuestionID": qid_email,
            "QuestionType": "TE",
            "QuestionDescription": "Contact email",
            "Selector": "SL",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation": {
                "Settings": {
                    "ForceResponse": "OFF",
                    "Type": "ContentType",
                    "ContentType": "ValidEmail"
                }
            },
            "Language": [],
            "DataExportTag": qid_email,
        }
    })
    welcome_block_elements.append({"Type": "Question", "QuestionID": qid_email})
    
    # Create the welcome block
    welcome_block = {
        "Type": "Standard",
        "SubType": "",
        "Description": "Welcome",
        "ID": welcome_block_id,
        "BlockElements": welcome_block_elements
    }
    blocks.append(welcome_block)

    # Create "Who are you?" intro question
    intro_block_id = "BL_Intro"
    intro_block_elements = []
    
    qid_intro = new_qid()
    questions.append({
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": qid_intro,
        "SecondaryAttribute": "Who are you?",
        "TertiaryAttribute": None,
        "Payload": {
            "QuestionText": "Who are you?",
            "DefaultChoices": False,
            "DataExportTag": f"Q{qid_counter}",
            "QuestionType": "MC",
            "Selector": "SAVR",
            "SubSelector": "TX",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": "Who are you?",
            "Choices": {
                "1": {"Display": "undergraduate student"},
                "2": {"Display": "graduate student"},
                "3": {"Display": "faculty member"},
                "4": {"Display": "industry professional (which industry?)", "TextEntry": "true"},
                "5": {"Display": "other", "TextEntry": "true"}
            },
            "ChoiceOrder": [1, 2, 3, 4, 5],
            "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
            "GradingData": [],
            "Language": [],
            "NextChoiceId": 6,
            "NextAnswerId": 1,
            "QuestionID": qid_intro
        }
    })
    intro_block_elements.append({"Type": "Question", "QuestionID": qid_intro})
    
    # Create combined "Knowledge" question (AI + Publishing)
    # This question uses the same 7-point scale as book questions and conforms to QUESTION_STYLE
    KNOWLEDGE_QUESTIONS = [
        "Artificial intelligence",
        "Academic/textbook publishing"
    ]
    
    qid_knowledge = new_qid()
    
    if QUESTION_STYLE == "slider":
        knowledge_payload = {
            "QuestionText": "How knowledgeable are you about the following?",
            "QuestionID": qid_knowledge,
            "QuestionType": "Slider",
            "QuestionDescription": "Knowledge self-assessment",
            "Selector": "HSLIDER",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {
                "QuestionDescriptionOption": "UseText",
                "CSSliderMin": 1,
                "CSSliderMax": 7,
                "GridLines": 6,
                "SnapToGrid": True,
                "NumDecimals": "0",
                "CustomStart": False,
                "SliderStartPositions": {
                    "1": 0.5,
                    "2": 0.5
                },
                "NotApplicable": False,
                "MobileFirst": True
            },
            "Choices": {
                str(i): {"Display": q} for i, q in enumerate(KNOWLEDGE_QUESTIONS, start=1)
            },
            "ChoiceOrder": [1, 2],
            "Validation": {"Settings": {"Type": "None"}},
            "Language": [],
            "NextChoiceId": 3,
            "NextAnswerId": 1,
            "Labels": BOOK_QUESTION_LABELS,
            "Answers": {
                "1": {"Display": 1},
                "2": {"Display": 2},
                "3": {"Display": 3},
                "4": {"Display": 4},
                "5": {"Display": 5},
                "6": {"Display": 6},
                "7": {"Display": 7}
            },
            "DataExportTag": qid_knowledge,
        }
    else:  # radio
        knowledge_payload = {
            "QuestionText": "How knowledgeable are you about the following?",
            "QuestionID": qid_knowledge,
            "QuestionType": "Matrix",
            "QuestionDescription": "Knowledge self-assessment",
            "Selector": "Likert",
            "SubSelector": "SingleAnswer",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {
                "QuestionDescriptionOption": "UseText",
                "TextPosition": "inline",
                "ChoiceColumnWidth": 25,
                "RepeatHeaders": "none",
                "WhiteSpace": "OFF",
                "MobileFirst": True,
            },
            "Choices": {
                str(i): {"Display": q} for i, q in enumerate(KNOWLEDGE_QUESTIONS, start=1)
            },
            "ChoiceOrder": [str(i) for i in range(1, len(KNOWLEDGE_QUESTIONS) + 1)],
            "Answers": BOOK_QUESTION_LABELS,
            "AnswerOrder": list(range(1, 8)),
            "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
            "Language": [],
            "NextChoiceId": len(KNOWLEDGE_QUESTIONS) + 1,
            "NextAnswerId": 8,
            "ChoiceDataExportTags": False,
            "DataExportTag": qid_knowledge,
        }
    
    questions.append({
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": qid_knowledge,
        "SecondaryAttribute": "Knowledge self-assessment",
        "TertiaryAttribute": None,
        "Payload": knowledge_payload,
    })
    intro_block_elements.append({"Type": "Question", "QuestionID": qid_knowledge})
    
    # Create Intro block with demographic questions

    intro_block = {
        "Type": "Standard",
        "SubType": "",
        "Description": "Intro",
        "ID": intro_block_id,
        "BlockElements": intro_block_elements
    }

    # Trash block with a dummy question
    qid_trash = new_qid()
    questions.append({
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": qid_trash,
        "SecondaryAttribute": "Click to write the question text",
        "TertiaryAttribute": None,
        "Payload": {
            "QuestionText": "Click to write the question text",
            "DataExportTag": "Q1",
            "QuestionType": "MC",
            "Selector": "SAVR",
            "SubSelector": "TX",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": "Click to write the question text",
            "Choices": {
                "1": {"Display": "Click to write Choice 1"},
                "2": {"Display": "Click to write Choice 2"},
                "3": {"Display": "Click to write Choice 3"}
            },
            "ChoiceOrder": ["1", "2", "3"],
            "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
            "Language": [],
            "NextChoiceId": 4,
            "NextAnswerId": 1,
            "QuestionID": qid_trash
        }
    })
    
    trash_block = {
        "Type": "Trash", 
        "Description": "Trash / Unused Questions", 
        "ID": "BL_Trash", 
        "BlockElements": [{"Type": "Question", "QuestionID": qid_trash}]
    }
    
    # Final blocks list: Trash (1), Content (2), Intro (3)
    final_blocks = [trash_block] + blocks + [intro_block]
    
    # Convert blocks list to dict with numeric string keys (Qualtrics format)
    blocks_dict = {str(i): block for i, block in enumerate(final_blocks, start=1)}
    
    # Build Flow entries
    # Flow: Welcome block first, then Intro block, then Content block (Trash is not in flow)
    flow = []
    
    # Add Welcome block first
    flow.append({"ID": welcome_block_id, "Type": "Standard", "FlowID": "FL_4"})
    # Add Intro block
    flow.append({"ID": intro_block_id, "Type": "Standard", "FlowID": "FL_3"})
    # Add Content block (book ratings)
    flow.append({"ID": content_block_id, "Type": "Standard", "FlowID": "FL_2"})

    qsf = {
        "SurveyEntry": {
            "SurveyID": survey_id,
            "SurveyName": "Survey: Textbook Proposal Evaluation",
            "SurveyDescription": SURVEY_DESCRIPTION,
            "SurveyOwnerID": "UR_REDACTED",
            "SurveyBrandID": "REDACTED",
            "DivisionID": None,
            "SurveyLanguage": "EN",
            "SurveyActiveResponseSet": "RS_4Jc8vaZfTFlNYEe",
            "SurveyStatus": "Inactive",
            "SurveyStartDate": "0000-00-00 00:00:00",
            "SurveyExpirationDate": "0000-00-00 00:00:00",
            "SurveyCreationDate": "2025-12-03 21:37:48",
            "CreatorID": "UR_REDACTED",
            "LastModified": "2025-12-03 21:41:44",
            "LastAccessed": "0000-00-00 00:00:00",
            "LastActivated": "0000-00-00 00:00:00",
            "Deleted": None,
        },
        "SurveyElements": [
            {
                "SurveyID": survey_id,
                "Element": "BL",
                "PrimaryAttribute": "Survey Blocks",
                "SecondaryAttribute": None,
                "TertiaryAttribute": None,
                "Payload": blocks_dict  # Use dict instead of list
            },
            {
                "SurveyID": survey_id,
                "Element": "FL",
                "PrimaryAttribute": "Survey Flow",
                "SecondaryAttribute": None,
                "TertiaryAttribute": None,
                "Payload": {
                    "Flow": flow,
                    "Properties": {"Count": len(flow) + 1},  # +1 for the root element
                    "FlowID": "FL_1",
                    "Type": "Root"
                }
            },
            {
                "SurveyID": survey_id,
                "Element": "PL",
                "PrimaryAttribute": "Preview Link",
                "SecondaryAttribute": None,
                "TertiaryAttribute": None,
                "Payload": {"PreviewType": "Brand", "PreviewID": "a433d081-18b4-4419-825a-029711039d70"}
            },
            {
                "SurveyID": survey_id,
                "Element": "PROJ",
                "PrimaryAttribute": "CORE",
                "SecondaryAttribute": None,
                "TertiaryAttribute": "1.1.0",
                "Payload": {"ProjectCategory": "CORE", "SchemaVersion": "1.1.0"}
            },
            {
                "SurveyID": survey_id,
                "Element": "QC",
                "PrimaryAttribute": "Survey Question Count",
                "SecondaryAttribute": str(len(questions)),
                "TertiaryAttribute": None,
                "Payload": None
            },
            {
                "SurveyID": survey_id,
                "Element": "RS",
                "PrimaryAttribute": "RS_4Jc8vaZfTFlNYEe",
                "SecondaryAttribute": "Default Response Set",
                "TertiaryAttribute": None,
                "Payload": None
            },
            {
                "SurveyID": survey_id,
                "Element": "SCO",
                "PrimaryAttribute": "Scoring",
                "SecondaryAttribute": None,
                "TertiaryAttribute": None,
                "Payload": {
                    "ScoringCategories": [],
                    "ScoringCategoryGroups": [],
                    "ScoringSummaryCategory": None,
                    "ScoringSummaryAfterQuestions": 0,
                    "ScoringSummaryAfterSurvey": 0,
                    "DefaultScoringCategory": None,
                    "AutoScoringCategory": None
                }
            },
            {
                "SurveyID": survey_id,
                "Element": "SO",
                "PrimaryAttribute": "Survey Options",
                "SecondaryAttribute": None,
                "TertiaryAttribute": None,
                "Payload": {
                    "BackButton": "true",
                    "SaveAndContinue": "true",
                    "SurveyProtection": "PublicSurvey",
                    "BallotBoxStuffingPrevention": "false",
                    "AnonymizeResponse": "No",
                    "NoIndex": "Yes",
                    "SecureResponseFiles": "true",
                    "SurveyExpiration": "None",
                    "SurveyTermination": "DefaultMessage",
                    "Header": "",
                    "Footer": "",
                    "ProgressBarDisplay": "VerboseText",
                    "PartialData": "+1 week",
                    "ValidationMessage": "",
                    "PreviousButton": "Back",
                    "NextButton": "Next",
                    "SurveyTitle": "Survey: Textbook Proposal Evaluation",
                    "SkinLibrary": "REDACTED",
                    "SkinType": "component",
                    "Skin": {
                        "brandingId": None,
                        "templateId": "*simple",
                        "overrides": {
                            "colors": {"primary": "#ab0001"},
                            "questionText": {"size": "16px"},
                            "answerText": {"size": "15px"}
                        }
                    },
                    "NewScoring": 1,
                    "SurveyMetaDescription": SURVEY_DESCRIPTION,
                    "SurveyName": "Survey: Textbook Proposal Evaluation"
                }
            },
        ] + questions + [
            {
                "SurveyID": survey_id,
                "Element": "STAT",
                "PrimaryAttribute": "Survey Statistics",
                "SecondaryAttribute": None,
                "TertiaryAttribute": None,
                "Payload": {
                    "MobileCompatible": False,
                    "ID": "Survey Statistics"
                }
            }
        ]
    }
    return qsf

def main():
    setup_logging(log_name="generate_survey", log_level="INFO")
    
    csv_path = Path(__file__).parent / "selected_books.csv"
    if not csv_path.exists():
        LOGGER.error(f"CSV not found: {csv_path}. Please run select_survey_books.py first.")
        return

    LOGGER.info(f"Reading books from {csv_path}...")
    books = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            books.append({
                "title": row.get("title", ""),
                "abstract": row.get("abstract", "")
            })
            
    LOGGER.info(f"Loaded {len(books)} books.")

    # Generate Survey Content
    survey_content = format_qualtrics_txt(books)
    
    output_file = Path(__file__).parent / "survey.txt"
    with output_file.open("w") as f:
        f.write(survey_content)
    
    # Generate QSF export
    qsf_out = Path(__file__).parent / "survey.qsf"
    qsf_payload = build_qsf(books)
    with qsf_out.open("w") as f:
        json.dump(qsf_payload, f, indent=2)
        
    LOGGER.info(f"Survey written to {output_file}")
    LOGGER.info(f"QSF export written to {qsf_out}")

if __name__ == "__main__":
    main()
