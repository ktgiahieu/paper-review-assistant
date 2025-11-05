from typing import Optional

class ConsensusExtractionPrompt:
    prompt = """
    You will be provided with the review and rebuttal process for a research paper, as well as its content in markdown format. Your task is to analyze the dialogue and identify only the most crucial flaws that the authors acknowledge and must address for this paper to be publishable.
**Rules:**
1.  Focus strictly on significant weaknesses in methodology, statistical rigor, or experimental scope that impact the paper's core claims. Ignore minor points like typos or grammatical errors. If there is no specific section for weaknesses, obtain the crucial flaws based on the main idea being discussed between the reviewers and the authors.
2.  A flaw should only be included in the final list if it meets this specific "consensus" criteria:
    * A reviewer raises it as a major weakness, question, or limitation that needs to be adressed.
    * The authors' reply confirms its importance by either:
        a) Promising to add the required analysis, data, or clarification to the final, camera-ready version of the paper.
        b) Explicitly agreeing that it is a major limitation of the current study's scope.
3.  Do not include issues that the authors state will be addressed in separate "follow-up work" unless it is also clearly acknowledged as a critical limitation of the current paper.
4.  Your output MUST be a JSON object that conforms to the provided schema.**

Input review and rebuttal process:
<BEGIN REVIEW PROCESS>
{review_process}
<END REVIEW PROCESS>

Input paper:
<BEGIN PAPER>
{paper_text}
<END PAPER>
"""
    @staticmethod
    def generate_prompt_for_review_process(review_process: str, paper_text: str) -> str:
        return ConsensusExtractionPrompt.prompt.format(
            review_process=review_process,
            paper_text=paper_text
        )
        
        
class FlawInjectionPrompt:
    prompt = """You are an AI model creating a benchmark of flawed scholarly papers for a research experiment. Your objective is to modify a published research paper to re-introduce a specific flaw in a way that makes the paper's claims methodologically unsound.

The key is to make the paper *embody* the flaw, not just state it as a limitation. You must adopt the persona of an author who is either unaware of the flaw or actively downplaying it to make their results seem more significant. Stay true to the authors' original material and personas as much as possible, use scientific terms, avoid writing in an obviously different manner (e.g. "The phenomenon is visible to the naked eye"), ... to create an illusion that the paper is coherent, well-written and the introduced flaws are as hard to detect as possible.

**The Golden Rule:**
When rewriting, you must follow this golden rule: The flaw should **NOT** be acknowledged as a weakness in the "Limitations" or "Future Work" sections. Instead, the authors should either ignore it completely or misrepresent it as a minor point or even a strength (e.g., "Focusing on a single dataset allowed for a more controlled and in-depth analysis"). The conclusions and abstract must be written with unearned confidence, drawing broad claims from the now-flawed, narrow evidence.

---
Your Task (in two steps):
- Step 1: Thinking Block
First, during your thinking process (not in the final output), think step-by-step. Based on the **The Golden Rule**, analyze the paper and the flaw. Identify all sections that need to be changed to systemically and confidently embed the flaw. This will likely include the Abstract, Methods, Results, Discussion, and Conclusion. Critically, decide how you will rewrite the **Limitations** section to misdirect the reader from the core flaw.

- Step 2: Rewritten Sections Block (Output)
After the thinking block, generate the new, complete text for only the sections you identified.
Each rewritten section must begin with its original, exact markdown heading line.
The rewritten text must maintain the original paper's style but reflect the overconfident persona.
Your output MUST be a JSON object conforming to the provided schema.

---
The flaw to re-introduce is:
<BEGIN FLAW>
{flaw_description}
<END FLAW>

The original paper:
<BEGIN PAPER>
{original_paper_text}
<END PAPER>
"""
    @staticmethod
    def generate_prompt_for_flaw_injection(flaw_description: str, original_paper_text: str, last_error: Optional[str] = None) -> str:
        error_feedback_prompt = ""
        if last_error:
            error_feedback_prompt = f"""IMPORTANT CORRECTION: In the previous attempt, you provided a `target_heading` that was not found in the paper.
FAILED HEADING: "{last_error}"
Please carefully re-examine the full paper text provided below. Your new `target_heading` MUST EXACTLY MATCH an existing markdown heading in the text.


"""
            
        return error_feedback_prompt + FlawInjectionPrompt.prompt.format(
            flaw_description=flaw_description,
            original_paper_text=original_paper_text
        )
