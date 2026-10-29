"""Binder prompt factories and preset target sequences from the ESM cookbook."""

from __future__ import annotations
import random
from dataclasses import dataclass

from analysis.design.constants import MUTABLE_TOKEN, BinderPromptStr


@dataclass(frozen=True)
class PromptFactory:
    """A simple factory for making binder prompt strings."""

    name: str
    template: str
    length_ranges: dict[str, tuple[int, int]]
    is_antibody: bool

    def sample(self, seed: int) -> BinderPromptStr:
        random.seed(seed)
        return self.template.format(
            **{
                key: MUTABLE_TOKEN * random.randint(low, high)
                for key, (low, high) in self.length_ranges.items()
            }
        )


# fmt: off
BINDER_PROMPT_FACTORIES = {
    "minibinder": PromptFactory(name="minibinder", template="{seq}", length_ranges={"seq": (60, 200)}, is_antibody=False),
    "trastuzumab_framework_vhvl": PromptFactory(
        name="trastuzumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}YIHWVRQAPGKGLEWVARI{hcdr2}TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "atezolizumab_framework_vhvl": PromptFactory(
        name="atezolizumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}WIHWVRQAPGKGLEWVAWI{hcdr2}TYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "ocankitug_framework_vhvl": PromptFactory(
        name="ocankitug_framework_vhvl",
        template="QVQLVQSGAEVKKPGSSVKVSCKAS{hcdr1}WMHWVRQAPGQGLEWMGII{hcdr2}TSLNQKFQGRVTITADTSTSTAYMELSSLRSEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (8, 14), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    )
}


TARGET_SEQUENCES = {
    "cd45": "GSPGEPQIIFCRSEAAHQGVITWNPPQRSFHNFTLCYIKETEKDCLNLDKNLIKYDLQNLKPYTKYVLSLHAYIIAKVQRNGSAAMCHFTTKSAPPSQVWNMTVSMTSDNSMHVKCRPPRDRNGPHERYHLEVEAGNTLVRNESHKNCDFRVKDLQYSTDYTFKAYFHNGDYPGEPFILHHSTSY",
    "ctla4": "MHVAQPAVVLASSRGIASFVCEYASPGKATEVRVTVLRQADSQVTEVCAATYMMGNELTFLDDSICTGTSSGNQVNLTIQGLRAMDTGLYICKVELMYPPPYYLGIGNGTQIYVIDPE",
    "egfr": "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCV",
    "pd-l1": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
    "pdgfr": "GFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE",
}
# fmt: on
