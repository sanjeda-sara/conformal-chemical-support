"""Before making the repository public: move pilot-era outputs and one-off helpers out of the paths
the paper uses, so nobody mistakes a pilot number (seeds 0-9) for a confirmatory one.

Nothing is deleted.  Pilot tables and figures go to tables/pilot_archive/ and figures/pilot_archive/;
pilot-only scripts go to scripts/pilot/ (kept for transparency, not used by the paper); one-off
helpers and *.orig backups go to release_excluded/, which is added to .gitignore.  Uses `git mv` for
tracked files so history is kept.

usage: python scripts/prepare_release.py            # dry run: prints what would move
       python scripts/prepare_release.py --apply    # does it
Afterwards rerun Block 7 and scripts/check_cited_keys.py; both must still end in PASS.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RULES = [
    # (source dir, glob patterns, destination dir)
    ("tables", ["biology_*", "adaptive_vs_fixed*", "tursunbadalov_comparison.csv", "s1_simulation_*", "pilot_*",
                "descriptor_adjustment_coefficients.csv", "descriptor_adjustment_permutation.csv",
                "descriptor_adjustment_stratified.csv"], "tables/pilot_archive"),
    ("figures", ["pilot_*", "paper_fig2_hidden_failure.pdf", "paper_fig2_hidden_failure.png", "paper_fig4_nreq_dial.*"],
     "figures/pilot_archive"),
    ("scripts", ["run_biology.py", "make_paper_figures.py", "analyze_pilot.py", "run_nreq_sweep.py", "run_support_sweep.py"],
     "scripts/pilot"),
    ("scripts", ["apply_seed_filter_patch.py", "run_remaining.sh", "*.orig"], "release_excluded"),
    # the old appendix (replaced by supplement.tex) and its generated tables, which predate the confirmatory run
    ("paper/latex/generated", ["[A-RU-Z]_*.tex", "S1_panel*.tex"], "paper/latex/old_appendix/generated"),
    ("paper/latex", ["appendix.tex", "appendix.pdf", "make_appendix_tables.py", "main_draft_oct31.tex", "main_oct21.tex",
                     "theorem_packet.tex", "theorem_packet.pdf", "references.bib"],
     "paper/latex/old_appendix"),
]
KEEP = {"run_adaptive_vs_fixed.py"}          # imported by run_descriptor_adjustment.py; must stay in scripts/


def tracked(p):
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=REPO, capture_output=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    is_git = (REPO / ".git").exists()
    moves = []
    for src, pats, dst in RULES:
        for pat in pats:
            for f in sorted((REPO / src).glob(pat)):
                if f.is_file() and f.name not in KEEP:
                    moves.append((f, REPO / dst / f.name))
    if not moves:
        print("nothing to move (already prepared?)"); return
    for f, t in moves:
        print(f"{'MOVE' if a.apply else 'would move'}  {f.relative_to(REPO)}  ->  {t.relative_to(REPO)}")
        if a.apply:
            t.parent.mkdir(parents=True, exist_ok=True)
            if is_git and tracked(f):
                subprocess.run(["git", "mv", str(f), str(t)], cwd=REPO, check=True)
            else:
                shutil.move(str(f), str(t))
    if a.apply:
        gi = REPO / ".gitignore"
        lines = gi.read_text().splitlines() if gi.exists() else []
        if "release_excluded/" not in lines:
            gi.write_text("\n".join(lines + ["release_excluded/"]) + "\n")
        (REPO / "tables/pilot_archive/README.md").write_text(
            "Pilot-era outputs (seeds 0-9, or analyses superseded before the confirmatory run). "
            "Not used by the paper; kept for transparency.\n")
        od = REPO / "paper/latex/old_appendix"
        if od.exists():
            (od / "README.md").write_text("Superseded drafts and the old appendix (replaced by supplement.tex). Not used.\n")
        (REPO / "scripts/pilot/README.md").write_text(
            "Pilot-stage scripts (seeds 0-9). Not used by the paper; kept for transparency. "
            "Paths inside them assume the repository root as working directory.\n")
        print(f"\nmoved {len(moves)} files.  Now rerun Block 7 and scripts/check_cited_keys.py.")
    else:
        print(f"\n{len(moves)} files would move.  Rerun with --apply to do it.")


if __name__ == "__main__":
    main()
