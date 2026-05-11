import sys

from data.resumes import RESUME_DATASETS
from engine.ranker import ResumeMatchingEngine
from utils.excel_parser import parse_jd_excel


def main():
    xlsx_path = sys.argv[1] if len(sys.argv) > 1 else "job_descriptions_input.xlsx"

    print(f"Reading JDs from: {xlsx_path}")
    job_descriptions = parse_jd_excel(xlsx_path)

    if not job_descriptions:
        print("No JDs found in the Excel file. Please check the format.")
        sys.exit(1)

    engine = ResumeMatchingEngine(RESUME_DATASETS, job_descriptions)
    engine.run()


if __name__ == "__main__":
    main()