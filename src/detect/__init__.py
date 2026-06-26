"""Detection extension (handoff7 Step B): flag deceptive CoTs by PROCESS, not plausibility.

The replication showed holistic judges (validator V / a Trust-Score LLM) cannot flag the
backdoor's deception — triggered-wrong CoTs pass V ~98% (Vwrong) because each is a fluent,
well-formed chain whose answer is wrong by a SINGLE localized step error. This package
implements per-step process verification that targets exactly that localized error.
"""
