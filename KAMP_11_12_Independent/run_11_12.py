# 08 학습/검증 코드를 호출하지 않고 최종 산출물만 사용하여 11,12를 실행한다.
# 명시한 출력 경로 아래에만 CSV/JSON을 생성하며 08 파일은 변경하지 않는다.
from pathlib import Path
import argparse
import importlib.util

HERE = Path(__file__).resolve().parent

def run(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd(), help="KAMP 저장소 최상단")
    p.add_argument("--output", type=Path, default=None, help="산출물을 저장할 별도 폴더")
    args = p.parse_args()
    root = args.root.resolve()
    out = (args.output or (root / "11_12_results")).resolve()
    if out == root or out.is_relative_to(root / "08_Modeling") or out.is_relative_to(root / "07_Feature_Engineering"):
        raise ValueError("출력 경로를 08/07 내부로 정하지 마세요")
    a = run(HERE / "11_Process_Interpretation/11_01_Process_Interpretation.py", "stage11")
    b = run(HERE / "12_Optimization_Decision/12_01_Operation_Decision.py", "stage12")
    a.main(root, out / "11")
    b.main(root, out / "11", out / "12")
    print("\n완료. 원본 08 코드/결과를 변경하지 않았습니다.")

if __name__ == "__main__":
    main()
