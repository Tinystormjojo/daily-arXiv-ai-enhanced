from pydantic import BaseModel, Field

class Structure(BaseModel):
    tldr: str = Field(description="generate a too long; didn't read summary")
    motivation: str = Field(description="describe the motivation in this paper")
    method: str = Field(description="method of this paper")
    result: str = Field(description="result of this paper")
    conclusion: str = Field(description="conclusion of this paper")
    author_affiliations: str = Field(
        description=(
            "Organizations, affiliations, or employers of the authors, inferred ONLY from the "
            "given title and abstract. If the text does not mention or clearly imply them, state "
            "that affiliations cannot be determined from the abstract (do not guess or fabricate)."
        )
    )
    production_deployment: str = Field(
        description=(
            "Whether the paper claims real production or online deployment, live A/B tests, "
            "industrial adoption, or partnership, versus offline experiments, simulation, or "
            "benchmarks on public datasets only. Say explicitly if the abstract is silent."
        )
    )
    generative_recommendation: str = Field(
        description=(
            "Whether the work is substantively related to end-to-end generative recommendation "
            "(generative models as the core of recommendation from user/item signals to final "
            "recommended items or content). Answer yes/no/unclear, explain the relationship, and "
            "point to where it appears (problem formulation, model architecture, training, or "
            "evaluation). If not related, give a brief reason."
        )
    )