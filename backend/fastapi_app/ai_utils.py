import os
import time
import google.generativeai as genai
from google.generativeai import types
import uuid
import sys
import tempfile
import shutil
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, unquote,urlunparse


# def get_signed_url(public_url: str, supabase_url: str, supabase_key: str, expires_in: int = 3600) -> str:
#     """
#     Get a signed URL from Supabase storage.
#     """
#     if "token=" in public_url:
#         return public_url
    
#     parsed = urlparse(public_url)
#     path = parsed.path
    
#     marker = "/storage/v1/object/public/"
#     idx = path.find(marker)
#     if idx == -1:
#         raise ValueError("URL does not contain '/storage/v1/object/public/'")
    
#     # Extract bucket and file path
#     after_marker = path[idx + len(marker):]
#     parts = after_marker.split('/', 1)
#     if len(parts) != 2:
#         raise ValueError("Could not parse bucket and path from URL")
    
#     bucket_name, file_path = parts
    
#     # Request signed URL from Supabase
#     sign_url = f"{supabase_url.rstrip('/')}/storage/v1/object/sign/{bucket_name}/{file_path}"
    
#     headers = {
#         "apikey": supabase_key,
#         "Authorization": f"Bearer {supabase_key}",
#         "Content-Type": "application/json"
#     }
    
#     payload = {"expiresIn": expires_in}
    
#     resp = requests.post(sign_url, json=payload, headers=headers, timeout=10)
#     if resp.status_code != 200:
#         raise Exception(f"Failed to get signed URL: {resp.status_code} {resp.text}")
    
#     signed_path = resp.json().get("signedURL")
#     if not signed_path:
#         raise Exception("No signedURL in response")
    
#     # Construct full URL
#     return f"{supabase_url.rstrip('/')}{signed_path}"

def setup_auth():
    """Sets up authentication for the Gemini API by checking for an env var."""
    try:
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        print("Authentication configured using GOOGLE_API_KEY.")
    except KeyError:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during authentication setup: {e}")
        sys.exit(1)


def grade_student_answer(rubric_text: str, student_answer: str, model_name: str = "gemini-2.5-flash"):
    model = genai.GenerativeModel(model_name=model_name)
    grading_prompt = f"""
    You are an expert teacher grading a student's submission.

    Rubric (each question's grading criteria):
    {rubric_text}

    Student's Answers:
    {student_answer}

    ---
    TASK:
    1. Identify each question number (like 1.a, 1.b, etc.).
    2. For each question, use the rubric to decide a numeric score.
    3. Provide a short reason for why that score fits the rubric.
    4. Suggest how the student can improve.

    Only use numeric scores listed in the rubric. Do not invent new scales.

    OUTPUT FORMAT:
    {{
      "results": [
        {{
          "question": "1.a",
          "score": <number>,
          "reason": "<reason based on rubric>",
          "improvement": "<how to improve>"
        }},
        ...
      ],
      "overall_feedback": "<overall comment summarizing performance>"
      "total_score":"<The total score achieved by the student>"
    }}
    """
    
    # Add safety settings to allow educational content
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
    
    try:
        response = model.generate_content(
            grading_prompt,
            generation_config=types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=100000
            ),
            safety_settings=safety_settings
        )
        
        # Check if response was blocked
        if not response.candidates:
            return {
                "error": "Response blocked by safety filters",
                "finish_reason": "SAFETY",
                "detail": "No candidates returned"
            }
        
        # Check finish reason
        candidate = response.candidates[0]
        if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
            finish_reasons = {
                0: "FINISH_REASON_UNSPECIFIED",
                1: "STOP",
                2: "SAFETY",
                3: "RECITATION",
                4: "OTHER"
            }
            return {
                "error": "Response not completed normally",
                "finish_reason": finish_reasons.get(candidate.finish_reason, "UNKNOWN"),
                "safety_ratings": [
                    {
                        "category": rating.category,
                        "probability": rating.probability
                    } for rating in candidate.safety_ratings
                ]
            }
        
        return response.text
        
    except Exception as e:
        return {
            "error": "Exception during generation",
            "detail": str(e)
        }

def transcribe_pdf_from_path(pdf_path: str, system_prompt: str, model_name: str = "gemini-2.5-flash"):
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
    except Exception as e:
        return f"Error: Could not instantiate model {model_name}."

    pdf_file = None
    try:
        pdf_file = genai.upload_file(
            path=pdf_path,
            display_name=os.path.basename(pdf_path)
        )

        while pdf_file.state.name == "PROCESSING":
            time.sleep(10)
            pdf_file = genai.get_file(name=pdf_file.name)

        if pdf_file.state.name != "ACTIVE":
            raise Exception(f"File processing failed. Final state: {pdf_file.state.name}")

        response = model.generate_content(
            [pdf_file, "Please transcribe this document following all instructions."],
            generation_config=types.GenerationConfig(
                max_output_tokens=15000,
                temperature=0.0
            )
        )
        text_output = response.text

    except Exception as e:
        text_output = f"Error: {e}"
    finally:
        if pdf_file:
            try:
                genai.delete_file(name=pdf_file.name)
            except Exception:
                pass

    return text_output
def construct_full_storage_url(file_path: str, supabase_url: str, bucket_name: str) -> str:
    """Construct full Supabase storage URL from various input formats."""
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return file_path
    
    file_path = file_path.lstrip("/")
    
    if not file_path.startswith(f"{bucket_name}/"):
        file_path = f"{bucket_name}/{file_path}"
    
    return f"{supabase_url.rstrip('/')}/storage/v1/object/public/{file_path}"


def get_signed_url(file_path: str, supabase_url: str, supabase_key: str, 
                   bucket_name: str, expires_in: int = 604800) -> str:
    """Get signed URL from Supabase storage."""
    full_url = construct_full_storage_url(file_path, supabase_url, bucket_name)
    
    if "token=" in full_url:
        return full_url
    
    parsed = urlparse(full_url)
    path = parsed.path
    
    marker = "/storage/v1/object/public/"
    idx = path.find(marker)
    if idx == -1:
        raise ValueError(f"URL malformed: {full_url}")
    
    after_marker = path[idx + len(marker):]
    parts = after_marker.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse bucket/path: {full_url}")
    
    bucket, file_path_clean = parts
    
    sign_url = f"{supabase_url.rstrip('/')}/storage/v1/object/sign/{bucket}/{file_path_clean}"
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }
    
    payload = {"expiresIn": expires_in}
    
    print(f"   Requesting signed URL from: {sign_url}")
    resp = requests.post(sign_url, json=payload, headers=headers, timeout=10)
    
    if resp.status_code != 200:
        raise Exception(f"Sign request failed: {resp.status_code} - {resp.text}")
    
    data = resp.json()
    signed_path = data.get("signedURL").split('?token')[-1]
    print(data)
    if not signed_path:
        raise Exception(f"No signedURL in response: {data}")
    
    return sign_url+"?token"+signed_path

def grade_submissions_for_assignment(assignment_id: str, assignment_idea: str, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch submissions for an assignment from Supabase, transcribe and grade each one.

    Parameters:
      - assignment_id: ID used in the `submissions` table to filter rows.
      - assignment_idea: rubric or assignment description text passed to grader.
      - supabase_url/supabase_key: optional overrides for environment variables.

    Returns a dict: {"count": n, "results": [...] }
    """
    # Ensure Gemini auth configured
    setup_auth()

    SUPABASE_URL = supabase_url or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    SUPABASE_ROLE = os.environ.get("SUPABASE_ROLE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase URL or key not provided via args or environment variables")

    rest_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/submissions"
    params = {"select": "*", "assignment_id": f"eq.{assignment_id}"}
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }

    resp = requests.get(rest_url, params=params, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch submissions from Supabase: {resp.status_code} {resp.text}")

    submissions = resp.json()

    results: List[Dict[str, Any]] = []

    PROMPT_ANSWERSCRIPT = (
        "You are an expert transcriptionist specializing in handwritten documents."
        "Transcribe the attached PDF, which contains handwritten questions and answers."
        "Your task is to produce a clean, plain-text version of the content."
        "Follow these rules precisely:"
        "1. Preserve the question and answer (Q&A) format."
        "2. Start each question with the prefix 'Question:' on a new line."
        "3. Start each answer with the prefix 'Answer:' on a new line."
        "4. For any handwritten math, transcribe it into clear, readable LaTeX format (e.g., $E = mc^2$, $\\frac{a}{b}$)."
    )

    tmpdir = tempfile.mkdtemp(prefix="submissions_")
    try:
        for sub in submissions:
            try:
                user_id = sub.get("user_id")
                file_url = sub.get("file_url")
                submission_id = sub.get("id")
                
                print(f"\n📄 Processing submission:")
                print(f"   Submission ID: {submission_id}")
                print(f"   User ID: {user_id}")
                print(f"   Raw file_url: '{file_url}'")
                
                if not file_url:
                    results.append({
                        "submission_id": submission_id,
                        "user_id": user_id,
                        "status": "skipped",
                        "reason": "no public file_url present"
                    })
                    continue

                # Try different approaches
                
                # Approach 1: Direct download (if public bucket)
                print(f"\n🔄 Attempt 1: Direct download")
                try:
                    direct_url = construct_full_storage_url(file_url, SUPABASE_URL, "submissions")
                    print(f"   Direct URL: {direct_url}")
                    
                    direct_resp = requests.get(direct_url, timeout=10)
                    print(f"   Direct response status: {direct_resp.status_code}")
                    
                    if direct_resp.status_code == 200:
                        print(f"   ✅ Direct download successful!")
                        local_name = os.path.join(tmpdir, f"{uuid.uuid4()}_{os.path.basename(file_url)}")
                        with open(local_name, "wb") as f:
                            f.write(direct_resp.content)
                        
                        student_text = transcribe_pdf_from_path(local_name, PROMPT_ANSWERSCRIPT)
                        grading = grade_student_answer(assignment_idea, student_text)
                        
                        results.append({
                            "submission_id": submission_id,
                            "user_id": user_id,
                            "status": "graded",
                            "grading": grading
                        })
                        continue
                except Exception as e:
                    print(f"   ❌ Direct download failed: {e}")

                # Approach 2: Signed URL
                print(f"\n🔄 Attempt 2: Signed URL")
                try:
                    signed_url = get_signed_url(file_url, SUPABASE_URL, SUPABASE_KEY, "submissions")
                    print(f"   Signed URL: {signed_url}")
                    
                    signed_resp = requests.get(signed_url, timeout=10)
                    print(f"   Signed response status: {signed_resp.status_code}")
                    
                    if signed_resp.status_code == 200:
                        print(f"   ✅ Signed download successful!")
                        local_name = os.path.join(tmpdir, f"{uuid.uuid4()}_{os.path.basename(file_url)}")
                        with open(local_name, "wb") as f:
                            f.write(signed_resp.content)
                        
                        student_text = transcribe_pdf_from_path(local_name, PROMPT_ANSWERSCRIPT)
                        grading = grade_student_answer(assignment_idea, student_text)
                        print(grading)
                        results.append({
                            "submission_id": submission_id,
                            "user_id": user_id,
                            "status": "graded",
                            "grading": grading
                        })
                        continue
                    else:
                        print(f"   ❌ Signed download failed: {signed_resp.text[:200]}")
                except Exception as e:
                    print(f"   ❌ Signed URL failed: {e}")

                # If both failed
                results.append({
                    "submission_id": submission_id,
                    "user_id": user_id,
                    "status": "download_failed",
                    "detail": "Both direct and signed URL approaches failed"
                })
                
            except Exception as e:
                print(f"❌ Error processing submission: {e}")
                results.append({
                    "submission_id": sub.get("id"),
                    "status": "error",
                    "detail": str(e)
                })
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    return {"count": len(results), "results": results}
