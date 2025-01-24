import sys
import time  # Import the time module

print("sys.prefix:", sys.prefix)

from google.auth.transport.requests import Request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
import pickle
import base64
import email
from email.mime.text import MIMEText
import json
import warnings

# Serper Tool Imports - Updated to GoogleSerperAPIWrapper and load_tools
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, AgentType, load_tools

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
print("Attempting to load environment variables...")
load_dotenv()
print("Environment variables loaded (hopefully!).")

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def test_llm():
    """Test LLM connection - now for Gemini"""
    try:
        print("Testing LLM connection (Gemini)...")
        print("Trying to get GOOGLE_API_KEY from environment in test_llm...")
        google_api_key = os.getenv('GOOGLE_API_KEY')
        print(f"Value of google_api_key retrieved in test_llm: {google_api_key}")
        if not google_api_key:
            print("GOOGLE_API_KEY is empty or None in test_llm.")
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        else:
            print("GOOGLE_API_KEY found and seems valid.")

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temperature=0.7
        )

        response = llm.invoke([HumanMessage(content="What is 2+2? Answer with just the number.")])
        print("✓ LLM Connection Test Successful! (Gemini)")
        print(f"Test response (Gemini): {response}")
        return True
    except Exception as e:
        print(f"❌ LLM Connection Error (Gemini): {type(e).__name__}: {str(e)}")
        return False

def test_serper_tool():
    """Test GoogleSerperAPIWrapper connection using load_tools"""
    try:
        print("Testing GoogleSerperAPIWrapper using load_tools...")
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")

        tools = load_tools(["google-serper"], serper_api_key=serper_api_key)
        search_tool = tools[0] # Assuming google-serper is the first tool loaded
        result = search_tool.run("What is the capital of France?")
        print("✓ GoogleSerperAPIWrapper Test Successful using load_tools!")
        print(f"Test response from Google Serper: {result}")
        return True
    except Exception as e:
        print(f"❌ GoogleSerperAPIWrapper Test Error: {type(e).__name__}: {str(e)}")
        return False


class GmailAssistant:
    def __init__(self):
        print("Initializing Gmail Assistant...")
        self._initialize_llm()
        self._initialize_tools() # Initialize tools here - using load_tools now
        self._initialize_chains()
        self._initialize_agent() # Initialize agent here
        self._initialize_gmail_service()
        print("Gmail Assistant initialization complete!")

    def _initialize_llm(self):
        """Initialize the LLM - now using Gemini"""
        try:
            print("Starting LLM initialization (Gemini)...")
            print("Trying to get GOOGLE_API_KEY from environment...")
            google_api_key = os.getenv('GOOGLE_API_KEY')
            print(f"Value of google_api_key retrieved from env: {google_api_key}")
            if not google_api_key:
                print("GOOGLE_API_KEY is empty or None after retrieval.")
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            else:
                print("GOOGLE_API_KEY found and seems valid.")

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=google_api_key,
                temperature=0.7
            )
            # Test the LLM
            test_response = self.llm.invoke([HumanMessage(content="Test LLM - respond with 'OK'")])
            print(f"LLM test response (Gemini): {test_response}")

        except Exception as e:
            print(f"❌ LLM initialization failed (Gemini): {type(e).__name__}: {str(e)}")
            raise

    def _initialize_tools(self):
        """Initialize tools for the agent, including Google Serper Search using load_tools"""
        print("Initializing tools using load_tools...")
        try:
            serper_api_key = os.getenv("SERPER_API_KEY")
            if not serper_api_key:
                print("Warning: SERPER_API_KEY not found in environment variables. Web search will be disabled.")
                self.tools = [] # No tools if API key is missing
            else:
                self.tools = load_tools(["google-serper"], serper_api_key=serper_api_key) # Load Google Serper tool
                print("Google Serper Tool initialized using load_tools.")
        except Exception as e:
            print(f"❌ Tool initialization failed: {e}")
            self.tools = [] # Fallback to no tools in case of error
        print("Tools initialization complete.")


    def extract_email_body(self, message_payload):
        """Extract and clean email body content"""
        def decode_content(encoded_data):
            try:
                return base64.urlsafe_b64decode(encoded_data).decode('utf-8', errors='ignore')
            except Exception as e:
                print(f"Decoding error: {e}")
                return ''

        def clean_html(content):
            import re
            # Remove HTML tags
            clean = re.compile('<.*?>')
            text = re.sub(clean, '', content)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        body_parts = []

        try:
            if 'parts' in message_payload:
                for part in message_payload['parts']:
                    if part.get('mimeType') == 'text/plain':
                        if 'data' in part.get('body', {}):
                            text = decode_content(part['body']['data'])
                            if text.strip():
                                body_parts.append(text)
                    elif part.get('mimeType') == 'text/html':
                        if 'data' in part.get('body', {}):
                            html = decode_content(part['body']['data'])
                            if html.strip():
                                body_parts.append(clean_html(html))

            # If no parts found, try body directly
            if not body_parts and 'body' in message_payload and 'data' in message_payload['body']:
                content = decode_content(message_payload['body']['data'])
                if '<html' in content.lower():
                    body_parts.append(clean_html(content))
                else:
                    body_parts.append(content)

        except Exception as e:
            print(f"Body extraction error: {e}")
            return "Error extracting email body"

        return '\n'.join(body_parts).strip() or "No meaningful content found in email body."

    def _initialize_chains(self):
        """Initialize LangChain chains"""
        try:
            # Analyzer chain
            analyzer_template = """You are an insightful email analyzer tasked with understanding the complete content and context of emails. You have access to web search to look up information if needed.

            Carefully analyze this email:
            From: {sender}
            Subject: {subject}
            Body: {body}

            Utilize web search if you encounter unfamiliar terms or need more context to understand the email's content better.

            Provide a thorough analysis considering:
            - The main message and its implications
            - The sender's intent and tone
            - The quality and value of the content
            - Any actionable insights or key learnings

            Return your analysis as a detailed JSON object:
            {{
                "priority": "high/medium/low",
                "category": "request/information/action/system/notification/marketing/personal/other",
                "sentiment": "positive/neutral/negative",
                "summary": "detailed multi-sentence summary covering main points and implications",
                "key_points": ["detailed point 1", "detailed point 2", "detailed point 3", "detailed point 4"],
                "required_action": "immediate/follow-up/no-action",
                "content_value": "assessment of the content's value and relevance",
                "engagement_level": "high/medium/low - how engaging is the content"
            }}""" # Corrected JSON format for Gemini and added Serper instruction

            # Response chain
            response_template = """You are a thoughtful email responder who creates engaging, valuable responses to email content. You have analyzed the email and now need to draft a response. You also have access to web search to enhance your response if necessary.

            Original Email:
            From: {sender}
            Subject: {subject}
            Body: {body}

            Analysis: {analysis}

            Utilize web search to gather additional information if it helps you craft a more informed and valuable response.

            Task: Generate an engaging, detailed response that:
            1. Acknowledges the value of the content
            2. Shares specific insights or reflections based on the analysis and potentially web search results
            3. Adds to the discussion with relevant examples or thoughts
            4. Maintains a conversational, authentic tone
            5. Shows genuine interest in the topic

            Return your response as a JSON object:
            {{
                "response_text": "Write a detailed, multi-paragraph response that engages meaningfully with the content. Include specific references to the material, analysis, and any insights from web search. Maintain a natural, conversational tone. The response should be substantial enough to demonstrate real engagement with the ideas presented.",
                "tone": "formal/casual/professional/friendly",
                "follow_up_needed": true/false,
                "key_response_points": ["detailed point about your response 1", "detailed point about your response 2", "detailed point about your response 3"],
                "engagement_type": "how you're engaging with the content (e.g., 'expanding on ideas', 'sharing personal insights', 'discussing implications')"
            }}""" # Corrected JSON format for Gemini and added Serper instruction


            # Create prompts
            analyzer_prompt = ChatPromptTemplate.from_template(analyzer_template)
            response_prompt = ChatPromptTemplate.from_template(response_template)

            # Initialize chains
            self.analyzer_chain = LLMChain(llm=self.llm, prompt=analyzer_prompt)
            self.response_chain = LLMChain(llm=self.llm, prompt=response_prompt)

            print("✓ LangChain chains initialized")

        except Exception as e:
            print(f"Chain initialization error: {str(e)}")
            raise

    def _initialize_agent(self):
        """Initialize the Langchain Agent with error handling"""
        try:
            if not self.tools:
                print("Agent initialization skipped - no tools available")
                return

            self._agent = initialize_agent( # Renamed to self._agent to avoid shadowing
                self.tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,  # Add this to handle parsing errors
                verbose=True
            )
            print("Agent initialized successfully.")
            self.agent = self._agent # Assign to self.agent AFTER successful initialization
        except Exception as e:
            print(f"❌ Agent initialization error: {e}")
            self.agent = None # Ensure self.agent is set to None even if initialization fails

    def _initialize_gmail_service(self):
        """Initialize the Gmail service"""
        try:
            print("Setting up Gmail API...")
            creds = None

            if os.path.exists('token.pickle'):
                print("Loading existing credentials from token.pickle")
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    print("Refreshing expired credentials")
                    creds.refresh(Request())
                else:
                    print("Getting new credentials - please authenticate in your browser")
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)

                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)

            self.service = build('gmail', 'v1', credentials=creds)
            print("✓ Gmail API setup complete")

        except Exception as e:
            print(f"❌ Gmail service initialization failed: {type(e).__name__}: {str(e)}")
            raise

    def get_unread_emails(self):
        """Fetch unread emails from Gmail"""
        try:
            results = self.service.users().messages().list(
                userId='me',
                labelIds=['UNREAD'],
                maxResults=10
            ).execute()

            messages = results.get('messages', [])
            return messages
        except Exception as e:
            print(f"Error fetching emails: {str(e)}")
            return []

    def process_email(self, message_id):
        """Process a single email with comprehensive analysis"""
        try:
            # Fetch the full email message
            print(f"Fetching full email message for ID: {message_id}...") # DEBUG PRINT
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            print(f"Full email message fetched successfully for ID: {message_id}.") # DEBUG PRINT

            # Extract headers
            headers = message['payload']['headers']
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')

            # Extract body with better error handling
            print(f"Extracting email body for ID: {message_id}...") # DEBUG PRINT
            try:
                body = self.extract_email_body(message['payload'])
                if not body or body == "No meaningful content found in email body.":
                    print(f"Warning: No meaningful body content found for email {message_id}")
                    body = "Empty email body"
            except Exception as e:
                print(f"Body extraction error: {e}")
                body = "Error extracting email body"
            print(f"Email body extracted for ID: {message_id}. Length: {len(body)}") # DEBUG PRINT

            print("\n--- EMAIL PROCESSING DEBUG ---")
            print(f"Message ID: {message_id}")
            print(f"From: {sender}")
            print(f"Subject: {subject}")
            print(f"Body Length: {len(body)}")
            print("--- Body Preview (first 500 chars) ---")
            print(body[:500] + "..." if len(body) > 500 else body)
            print("--- END BODY PREVIEW ---\n")

            try:
                analysis_json = self._analyze_email_content(sender, subject, body)
                response_json = self._generate_email_response(sender, subject, body, analysis_json)

                return {
                    "analysis": analysis_json,
                    "response": response_json
                }

            except Exception as e:
                print(f"Processing error during analysis/response: {str(e)}")
                return self._create_processing_error_result(subject) # Use dedicated error result function

        except Exception as e:
            print(f"Error processing email {message_id}: {str(e)}")
            return None

    def _analyze_email_content(self, sender, subject, body):
        analysis_input = {"sender": sender, "subject": subject, "body": body}
        if self.agent:
            analysis_prompt_str = f"""
                    Analyze this email and return a structured JSON response:
                    From: {sender}
                    Subject: {subject}
                    Body: {body}

                    Required JSON format:
                    {{
                        "priority": "high/medium/low",
                        "category": "request/information/action/other",
                        "sentiment": "positive/neutral/negative",
                        "summary": "<detailed summary>",
                        "key_points": ["point1", "point2"],
                        "required_action": "immediate/follow-up/no-action",
                        "content_value": "<value assessment>",
                        "engagement_level": "high/medium/low"
                    }}
                    """
            analysis_result_str = self.agent.run(analysis_prompt_str)
            try:
                analysis_json = json.loads(analysis_result_str)
            except json.JSONDecodeError as e:
                print(f"❌ JSON Decode Error (Analysis Agent Output): {e}")
                analysis_json = self._create_error_analysis_json(subject, analysis_result_str)
            return analysis_json


    def _generate_email_response(self, sender, subject, body, analysis_json):
        response_input = {"sender": sender, "subject": subject, "body": body, "analysis": json.dumps(analysis_json)}
        if self.agent:
            response_prompt_str = f"""
                    Based on this analysis: {json.dumps(analysis_json)}

                    Write a personalized email response to:
                    From: {sender}
                    Subject: {subject}
                    Original message: {body}

                    Make the response specific to the content and context.
                    Return the response as a plain text string, NOT a JSON object.
                    """ # Modified prompt to request plain text

            response_result_str = self.agent.run(response_prompt_str)
            response_json = {"response_text": response_result_str} # Wrap as JSON
        else:
            response_output = self.response_chain.invoke(response_input)
            response_json = response_output # No parsing needed for direct chain
        return response_json


    def send_response(self, message_id, response_data):
        """Send email response with robust recipient handling"""
        try:
            # Parse response data
            try:
                # If response_data is a string, try to parse it as JSON
                if isinstance(response_data, str):
                    response_json = json.loads(response_data)
                else:
                    # If it's already a dict, use it directly
                    response_json = response_data

                # Extract response text, defaulting to the entire input if parsing fails
                final_response = response_json.get('response_text', str(response_data))
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, use the original input as the response
                final_response = str(response_data)

            # Get original message details
            original_message = self.service.users().messages().get(
                userId='me',
                id=message_id
            ).execute()

            # Robust recipient extraction
            def extract_valid_recipient(headers):
                """Extract a valid recipient email address"""
                # Try different header combinations
                recipient_headers = [
                    'Reply-To', 'From', 'Sender'
                ]

                for header_name in recipient_headers:
                    recipient = next((h['value'] for h in headers if h['name'] == header_name), None)
                    if recipient:
                        # Extract email address from header
                        import re
                        email_match = re.search(r'<([^>]+)>|(\S+@\S+)', recipient)
                        if email_match:
                            return email_match.group(1) or email_match.group(2)

                # Fallback if no valid recipient found
                return 'Unknown Sender'

            # Extract recipient
            recipient = extract_valid_recipient(original_message['payload']['headers'])

            # Get thread ID
            thread_id = original_message.get('threadId')

            # Create response email
            message = MIMEText(final_response)

            # Set recipient
            message['to'] = recipient

            # Set subject with 'Re:' prefix
            original_subject = next(
                (h['value'] for h in original_message['payload']['headers'] if h['name'] == 'Subject'),
                'No Subject'
            )
            message['subject'] = 'Re: ' + original_subject

            # Encode message
            raw = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode('utf-8')

            # Send response
            try:
                self.service.users().messages().send(
                    userId='me',
                    body={
                        'raw': raw,
                        'threadId': thread_id
                    }
                ).execute()
                print(f"✓ Sent response to email {message_id}")
            except Exception as send_error:
                print(f"❌ Error sending response: {send_error}")
                # Fallback sending (existing code) ...


        except Exception as e:
            print(f"❌ Comprehensive response sending error: {str(e)}")


    def _create_error_analysis_json(self, subject, raw_output):
        """Fallback JSON for analysis errors"""
        return {
            "priority": "low",
            "category": "other",
            "sentiment": "neutral",
            "summary": f"Failed to analyze email content properly for: {subject}. Raw output: {raw_output[:100]}...", # Limit raw output preview
            "key_points": ["Analysis parsing error occurred. Review raw output."],
            "required_action": "no-action",
            "content_value": "unknown",
            "engagement_level": "low"
        }

    def _create_error_response_json(self, subject, raw_output):
        """Fallback JSON for response errors"""
        return {
            "response_text": f"Apologies, I encountered an issue generating a detailed response for your email regarding: '{subject}'.  Raw output was problematic. I am sending a brief acknowledgment.",
            "tone": "professional",
            "follow_up_needed": False,
            "key_response_points": ["Generic acknowledgment due to error."],
            "engagement_type": "basic acknowledgment"
        }


    def _create_processing_error_result(self, subject):
        """Creates a unified error result for email processing failures"""
        return {
            "analysis": self._create_error_analysis_json(subject, "Processing failed before analysis."),
            "response": self._create_error_response_json(subject, "Processing failed before response generation.")
        }


def main():
    # Test LLM connection
    if not test_llm():
        print("Failed to connect to LLM (Gemini). Exiting...")
        return

    # Test Serper Tool connection
    if not test_serper_tool():
        print("Failed to connect to GoogleSerperAPIWrapper. Web search might be disabled. Continuing...")


    print("\nStarting Gmail automation...")
    try:
        assistant = GmailAssistant()
    except Exception as e:
        print(f"Failed to initialize Gmail Assistant: {str(e)}")
        return

    print("\nFetching unread emails...")
    unread = assistant.get_unread_emails()

    if not unread:
        print("No unread emails found.")
        return

    print(f"\nFound {len(unread)} unread emails.")

    for index, email in enumerate(unread, 1):
        print(f"\nProcessing email {index} of {len(unread)}")
        try:
            results = assistant.process_email(email['id'])
            if results and 'response' in results and results['response'].get('response_text'): # Check for response_text
                assistant.send_response(email['id'], results['response'])
                print(f"✓ Successfully processed and responded to email {email['id']}")
            else:
                print(f"⚠ No response generated for email {email['id']}")
        except Exception as e:
            print(f"❌ Error processing email {email['id']}: {str(e)}")
        time.sleep(5) # Add a 5-second delay between emails to help with rate limits

if __name__ == "__main__":
    main()