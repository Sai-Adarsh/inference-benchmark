use crate::Sample;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub ttft_ms: f64,
    pub total_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tpot_ms: Option<f64>,
}

impl Budget {
    pub fn validate(self) -> Result<(), &'static str> {
        if [
            Some(self.ttft_ms),
            Some(self.total_ms),
            self.estimated_tpot_ms,
        ]
        .into_iter()
        .flatten()
        .all(|limit| limit.is_finite() && limit > 0.0)
        {
            Ok(())
        } else {
            Err("SLO limits must be finite and positive")
        }
    }

    pub fn met(self, sample: &Sample) -> bool {
        sample.error.is_none()
            && sample
                .ttft_ms
                .is_some_and(|since_start| since_start <= self.ttft_ms)
            && sample.total_ms <= self.total_ms
            && self.estimated_tpot_ms.is_none_or(|limit| {
                sample.completion_tokens == Some(1)
                    || sample
                        .estimated_tpot_ms
                        .is_some_and(|observed| observed <= limit)
            })
    }

    pub fn label(self) -> String {
        let mut notes = format!(
            "SLO budgets per request are TTFT ≤ {} ms and total latency ≤ {} ms",
            self.ttft_ms, self.total_ms
        );
        if let Some(limit) = self.estimated_tpot_ms {
            notes.push_str(&format!(
                " with estimated TPOT ≤ {limit} ms for multi-token responses"
            ));
        }
        notes.push('.');
        notes
    }
}
