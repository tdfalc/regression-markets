from pathlib import Path


def save_figure(
    fig, savedir: Path, filename: str, dpi: int = 600, tight: bool = True
):
    if tight:
        fig.tight_layout()
    for extension in (".pdf", ".png"):
        fig.savefig(savedir / (filename + extension), dpi=dpi)
