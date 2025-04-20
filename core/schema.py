# core/schema.py

# --- JSON Schema for OCR ---
CNAM_FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "const": "Bulletin de remboursement des frais de soins"},
        "metadata": {
            "type": "object",
            "properties": {
                "identifiant_unique": {"type": "string"},
                "regime": {"type": "array", "items": {"type": "string", "enum": ["CNSS", "CNRPS", "Convention bilatérale"]}}
            },
            "required": ["identifiant_unique", "regime"],
            "additionalProperties": False
        },
        "assure_social": {
            "type": "object",
            "properties": {
                "nom": {"type": "string"},
                "prenom": {"type": "string"},
                "adresse": {
                    "type": "object",
                    "properties": {
                        "ligne_1": {"type": "string"},
                        "ligne_2": {"type": "string"},
                        "code_postal": {"type": "string"}
                    },
                    "required": ["ligne_1", "ligne_2", "code_postal"],
                    "additionalProperties": False
                }
            },
            "required": ["nom", "prenom", "adresse"],
            "additionalProperties": False
        },
        "Le_malade": {
            "type": "object",
            "properties": {
                "type_relation": {"type": "string", "enum": ["assure_social", "conjoint", "enfant", "ascendant"]},
                "prenom": {"type": "string"},
                "nom": {"type": "string"},
                "date_naissance": {"type": "string"},
                "telephone": {"type": "string"},
                "signature": {"type": "string"}
            },
            "required": ["type_relation", "prenom", "nom", "date_naissance", "telephone", "signature"],
            "additionalProperties": False
        }
    },
    "required": ["document_type", "metadata", "assure_social", "Le_malade"],
    "additionalProperties": False
}


# JSON Schema for claim evaluation including form data, purchase and tariff context
CLAIM_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        # Use the existing CNAM form schema for all extracted form fields
        "form_data": CNAM_FORM_SCHEMA,
        "purchase": {
            "type": "object",
            "description": "The one drug purchase to evaluate",
            "properties": {
                "code_acte":     {"type": "string", "const": "MED045"},
                "quantite":      {"type": "number"},
                "montant_total": {"type": "number"}
            },
            "required": ["code_acte", "quantite", "montant_total"],
            "additionalProperties": False
        },
        "tariff": {
            "type": "object",
            "description": "Reimbursement rules for the purchased drug",
            "properties": {
                "description":      {"type": "string"},
                "rate":             {"type": "number"},
                "ceiling_category": {"type": "string"}
            },
            "required": ["description", "rate", "ceiling_category"],
            "additionalProperties": False
        },
        "provider_valid": {
            "type": "object",
            "properties": {
                "answer": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["answer", "reason"],
            "additionalProperties": False
        },
        "codes_on_list": {
            "type": "object",
            "properties": {
                "answer": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["answer", "reason"],
            "additionalProperties": False
        },
        "dosage_matches": {
            "type": "object",
            "properties": {
                "answer": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["answer", "reason"],
            "additionalProperties": False
        },
        "line_item_reimbursements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code":   {"type": "string"},
                    "rate":   {"type": "number"},
                    "amount": {"type": "number"}
                },
                "required": ["code", "rate", "amount"],
                "additionalProperties": False
            }
        },
        "annual_ceiling_ok": {
            "type": "object",
            "properties": {
                "answer": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["answer", "reason"],
            "additionalProperties": False
        },
        "anomaly_flag": {
            "type": "object",
            "properties": {
                "answer": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["answer", "reason"],
            "additionalProperties": False
        },
        "final_decision": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "deny", "partial"]},
                "amount":   {"type": "number"}
            },
            "required": ["decision", "amount"],
            "additionalProperties": False
        }
    },
    "required": [
        "form_data",
        "purchase",
        "tariff",
        "provider_valid",
        "codes_on_list",
        "dosage_matches",
        "line_item_reimbursements",
        "annual_ceiling_ok",
        "anomaly_flag",
        "final_decision"
    ],
    "additionalProperties": False
}