import os
from ai_utils import grade_submissions_for_assignment  # your main script file

# ------------------------------
# 1. Configure environment
# ------------------------------
# Make sure these environment variables are set.
# You can either set them here manually or in your environment before running.

# os.environ["GEMINI_API_KEY"] = "<YOUR_GEMINI_API_KEY>"
# os.environ["SUPABASE_URL"] = "<YOUR_SUPABASE_URL>"
# os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "<YOUR_SUPABASE_SERVICE_ROLE_KEY>"

# ------------------------------
# 2. Define test parameters
# ------------------------------
ASSIGNMENT_ID = "02255cd5-897b-4eba-ae53-343ced526362"

# This is your rubric — the same format you mentioned earlier.
RUBRIC_TEXT = """
1.a) 0; not related to question,
     1; somewhat related to question,
     3; explanation proper but some gaps,
     5; proper clear explanation
2.a) 0; incorrect approach,
     2; partially correct logic,
     4; mostly correct but missing examples,
     5; perfect answer
"""

# ------------------------------
# 3. Run the grading pipeline
# ------------------------------
if __name__ == "__main__":
    print("🚀 Starting end-to-end grading test...")

    try:
        results = grade_submissions_for_assignment(
            assignment_id=ASSIGNMENT_ID,
            assignment_idea=RUBRIC_TEXT
        )

        print("\n✅ Finished grading!")
        print(f"Processed {results['count']} submissions.\n")

        for res in results["results"]:
            print(f"Submission ID: {res.get('submission_id')}")
            print(f"User ID: {res.get('user_id')}")
            print(f"Status: {res.get('status')}")
            print("----")

            if res.get("status") == "graded":
                print("Grading Output:\n", res.get("grading"))  # show first 1000 chars
                print("----\n")

    except Exception as e:
        print(f"❌ Error during grading: {e}")
