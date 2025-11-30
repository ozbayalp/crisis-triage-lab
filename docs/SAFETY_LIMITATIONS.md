# Safety & Limitations

> **Document Version:** 1.0  
> **Last Updated:** November 2024  
> **Classification:** CRITICAL — READ BEFORE USE

---

## If You Are In Crisis

**This software cannot help you.**

If you or someone you know is experiencing a mental health crisis or having thoughts of suicide:

| Region | Resource |
|--------|----------|
| **United States** | National Suicide Prevention Lifeline: **988** |
| **United States** | Crisis Text Line: Text **HOME** to **741741** |
| **United Kingdom** | Samaritans: **116 123** |
| **Canada** | Crisis Services Canada: **1-833-456-4566** |
| **Australia** | Lifeline: **13 11 14** |
| **International** | [findahelpline.com](https://findahelpline.com) |

**Please reach out to trained professionals who can provide real support.**

---

## Scope

### What This System Is

CrisisTriage AI is a **simulation and research tool** that:

- Demonstrates ML pipeline architecture for sensitive domains
- Processes synthetic or simulated inputs
- Produces outputs for research analysis only
- Runs entirely locally with privacy-first defaults

### What This System Is NOT

| This system is NOT... | Explanation |
|-----------------------|-------------|
| A crisis hotline | Cannot provide support or intervention |
| A diagnostic tool | Cannot assess mental health conditions |
| A clinical system | No medical or clinical validity |
| A replacement for professionals | Cannot substitute human judgment |
| A production service | Not designed for real users |
| A medical device | No regulatory approval of any kind |

---

## Key Risks

### Risk 1: Misinterpretation of Outputs

**Risk**: Users may interpret model outputs as valid clinical assessments.

**Reality**: 
- The model is trained on synthetic data
- Outputs have no clinical meaning
- "Risk levels" are arbitrary classifications for research
- There is no validated relationship between outputs and actual risk

**Mitigation**: 
- Explicit disclaimers throughout documentation
- UI warnings on all dashboards
- Research-only framing in all materials

---

### Risk 2: Overreliance on Model Predictions

**Risk**: Users may trust the model's outputs without appropriate skepticism.

**Reality**:
- The model will produce confident outputs even when wrong
- Edge cases and failure modes are numerous and untested
- No robustness guarantees exist
- The model cannot "know" when it's making errors

**Mitigation**:
- Confidence scores should not be interpreted as reliability
- All outputs require human interpretation
- No autonomous decision-making is supported

---

### Risk 3: Failure Modes

The model is known or expected to fail in these scenarios:

| Failure Mode | Example |
|--------------|---------|
| **Sarcasm** | "Oh great, another wonderful day" (distress masked as positivity) |
| **Coded language** | Euphemisms, indirect expressions, cultural idioms |
| **Brevity** | Very short inputs lack sufficient signal |
| **Context dependence** | Meaning that requires conversation history |
| **Cultural variation** | Non-Western expressions of distress |
| **Adversarial input** | Intentionally misleading text |
| **Multilingual** | Non-English or code-switched text |
| **Typos/errors** | Misspellings, autocorrect errors |

---

### Risk 4: Dual Use Concerns

**Risk**: The system could theoretically be misused for:
- Surveillance of individuals' mental states
- Automated decision-making about individuals
- Screening without consent

**Mitigations**:
- Local-only processing (no data exfiltration)
- No persistent storage by default
- No integration with external systems
- Research-only licensing

---

## Risk Mitigations in This Project

### Technical Mitigations

| Mitigation | Implementation |
|------------|----------------|
| **Local processing** | All inference runs on-device; no external APIs |
| **Ephemeral data** | Raw audio/transcripts not stored by default |
| **Anonymized logs** | Session IDs and content redacted from logs |
| **Bounded analytics** | Event history capped and auto-rotated |
| **No text snippets** | Analytics disabled from storing text by default |
| **Dummy services** | Development mode uses mock services |

### Documentation Mitigations

| Mitigation | Implementation |
|------------|----------------|
| **Model Card** | Explicit limitations and non-uses documented |
| **System Card** | Full system scope and constraints defined |
| **This Document** | Direct safety warnings and prohibitions |
| **UI Disclaimers** | Visible warnings on all interfaces |
| **README warnings** | Safety notices in all documentation |

### Architectural Mitigations

| Mitigation | Implementation |
|------------|----------------|
| **No authentication** | Cannot be mistaken for a real service |
| **No user accounts** | No personal data collection |
| **No external integrations** | Isolated from production systems |
| **Local-only default** | Not exposed to networks by default |

---

## Hard Prohibitions

The following uses are **strictly prohibited**:

### 1. Real Crisis Intervention

> ❌ **NEVER** use this system in any real-time crisis response context.

This includes:
- Actual crisis hotlines or chat services
- Emergency response systems
- Peer support platforms
- Any context with real individuals in distress

### 2. Clinical Deployment

> ❌ **NEVER** deploy this system in healthcare settings.

This includes:
- Hospitals or clinics
- Therapy or counseling practices
- Telehealth platforms
- Mental health apps for real users

### 3. User-Facing Services

> ❌ **NEVER** offer this system to real users as a support service.

This includes:
- Public-facing websites or apps
- Internal employee wellness tools
- Student support systems
- Any context where real humans interact expecting help

### 4. Automated Decisions

> ❌ **NEVER** use outputs for automated decision-making about individuals.

This includes:
- Triage without human review
- Risk scoring for access decisions
- Prioritization of real cases
- Any consequential automated action

### 5. Research on Real Populations

> ❌ **NEVER** use for research on real crisis data without proper oversight.

Any research involving real human subjects or real crisis data would require:
- Institutional Review Board (IRB) approval
- Appropriate ethical oversight
- Informed consent protocols
- Clinical collaborator involvement

---

## Liability Disclaimer

**THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.**

The authors and contributors:
- Make no claims about fitness for any purpose
- Accept no liability for any use or misuse
- Provide no guarantees about accuracy or reliability
- Disclaim all responsibility for any outcomes

**Users assume all risk associated with any use of this software.**

---

## Reporting Concerns

If you become aware of:
- Misuse of this system
- Deployment in prohibited contexts
- Safety issues not addressed here

Please contact: [Your Email]

---

## Acknowledgment

By using this software, you acknowledge that:

1. You have read and understood this document
2. You will not use the system for prohibited purposes
3. You understand the limitations and risks
4. You accept full responsibility for your use
5. You will not represent this as a clinical or production system

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | November 2024 | Initial safety documentation |

---

> **Final Reminder**: This is a research tool. It cannot help anyone in crisis. If you need support, please reach out to trained professionals or crisis services in your area.
