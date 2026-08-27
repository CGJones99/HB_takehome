Here's my submission for Problem A of the Huckberry Jr. AI Process Engineer take home assignment, taking the provided CSV and doing deterministic and LLM classification and returning a final JSON output.

Check the requirements.txt for required software and you'll need the customer_service_emails.csv to run the file. Code is entirely python and tooling used was Claude Code (Sonnet 5) as the execution layer and I called Haiku 4.5 via the api for text classification.

To skip straight to seeing the final output do the following. Spin up your environment according to the requirements.txt, add in the api key in a .env and name it API_KEY, ensure that customer_service_emails.csv is in the directory and run json_summary.py

Edge cases: 1) There were cases not in english, I chose to leave this alone due to time constraints and LLMs having some difficulty going from english prompting to ingesting another language depending on training data. That row is processed as normal with no special handling. 2) There were direct prompt injection attacks in HB-10277 and 10278 I chose to fight that head on with structured outputs and prompt guardrails to process as data so the tool even if it failed would return a malformed piece of data and not commands or a further attack. My deterministic processing rules actually stopped it from getting there but I wanted to run my LLM process on this to guarantee safety. 3) There were multiple submissisons and like #1 I chose to ignore knowing that the end goal was classification not bundling cases per customer identifier. They entered the normal process. 4) There were some cases that had no information in subject or body and I accounted for this in deterministic processing where if a response couldn't be classified at all it goes straight into other.

Structure:

Diagnose_csv pulls in the CSV and does some simple pandas operations to identify that the script isn't malformed and works. It provides a description of shape, and the datatype of each column to provide proof it's read.

Classify_csv then takes that csv and runs each row "scoring" combined subject and body of each ticket for the keywords and variants from the problem statement (refer to the listing for refund_return, shipping, and product_question). If only a single category is scored it is assigned a classification and hit count for that ticket number. If multiple categories score it is then classified as llm_review for further processing, as well as no categories hit rather than simply shipping off to other. Returns a quick summary table for operator to see that the classifications add to total rows.

llm_classify loads the API key, reruns Classify_csv to produce llm_review marked tickets, then sends only the subject and body of each to Haiku 4.5 to classify the text according to the 4 categories (refund_return through other), and return a confidence of low/med/high. Outputs are structured for pydantic schema and prompt instructs as data only processing and in the event of a detected injection to mark as other + low confidence. It returns a summary table so you can again check that the number of llm_review cases are processed.

Merge_summary brings the outputs of steps 2&3 together for operator validation that all cases were processed. Table becomes ticket#, classification, hit count (nullable), llm comment (nullable). Applies a rule where an LLM marked as other + low confidence as human_review.

I've added in human review as wherever the LLM operates we need people auditing the edge cases. I know that wasn't spec but I felt necessary system design.

json_summary does the full loop: calling Classify for the keyword pass, llm_classify for the llm pass on the remaining pages, merge for the final table build + human_review flags, and then prints a json summary of sum of cases by category adding to the total and then printing the ticket#, subject, and body for the ticket with the most deterministic hits with ties broken by which ticket number is first. Ticket numbering is what also decides other and human_review in the summary output. This is all printed in the console.
