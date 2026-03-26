from expert_verify_engine.models.schemas import EvidencePacket


def transform_evidence(
    raw_evidence: EvidencePacket,
    sensitivity: float = 1.0,
) -> EvidencePacket:
    adjusted_competence = {}
    for comp_name, comp_data in raw_evidence.competence.items():
        adjusted_competence[comp_name] = {
            "e_plus": comp_data.e_plus * sensitivity,
            "e_minus": comp_data.e_minus * sensitivity,
        }

    return EvidencePacket(
        competence=adjusted_competence,
        behavior=raw_evidence.behavior,
        notes=raw_evidence.notes,
    )
