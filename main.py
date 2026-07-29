"""程式進入點:python main.py [root] [--band Y1 Y2] [--cache N]

    D:\\ws\\gt_labeling> .venv\\Scripts\\activate
    (gt-labeling) D:\\ws\\gt_labeling> python main.py D:\\ws\\detect_stream\\out\\gt_sample

或不啟用 venv:

    uv run --project D:\\ws\\gt_labeling python main.py D:\\ws\\detect_stream\\out\\gt_sample
"""

from gt_labeling.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
