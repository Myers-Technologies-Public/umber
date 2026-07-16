//! Permission broker (docs/PLAN.md: "permission broker", D10 kernel boundary).
//!
//! **v1 semantics: deny everything.** The manifest's requested `[permissions]`
//! are stored and surfaced to the user, but the host ABI exposes no filesystem,
//! network, or process I/O at all — so there is nothing to grant. Every
//! capability check therefore returns [`Decision::Denied`] with a reason. The
//! broker exists now so the seam is real: when the ABI grows I/O imports (v2),
//! this is the single chokepoint every syscall passes through, and granting
//! becomes a matter of consulting the stored, user-approved permission lists.

use crate::manifest::Permissions;

/// A capability a module might request at an I/O boundary. String payloads are
/// the concrete target (a path, host, or program name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Capability {
    FsRead(String),
    FsWrite(String),
    Net(String),
    Exec(String),
}

impl Capability {
    fn kind(&self) -> &'static str {
        match self {
            Capability::FsRead(_) => "fs read",
            Capability::FsWrite(_) => "fs write",
            Capability::Net(_) => "net",
            Capability::Exec(_) => "exec",
        }
    }

    fn target(&self) -> &str {
        match self {
            Capability::FsRead(t)
            | Capability::FsWrite(t)
            | Capability::Net(t)
            | Capability::Exec(t) => t,
        }
    }
}

/// The outcome of a capability check. v1 only ever produces `Denied`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Granted (unreachable in v1; the variant exists for v2 wiring).
    Granted,
    /// Refused, with a human-readable reason.
    Denied(String),
}

impl Decision {
    pub fn is_granted(&self) -> bool {
        matches!(self, Decision::Granted)
    }
}

/// Per-module broker holding that module's requested (but ungranted)
/// permissions.
#[derive(Clone, Debug)]
pub struct PermissionBroker {
    requested: Permissions,
}

impl PermissionBroker {
    pub fn new(requested: Permissions) -> Self {
        Self { requested }
    }

    /// The permissions the manifest asked for (surfaced on the modules page).
    pub fn requested(&self) -> &Permissions {
        &self.requested
    }

    /// Check a capability. **Always denied in v1** — the ABI has no I/O, so the
    /// reason names the requested capability and the deny-by-default policy.
    pub fn check(&self, cap: &Capability) -> Decision {
        Decision::Denied(format!(
            "{} `{}` denied: v1 host grants no capabilities (deny-by-default, D10)",
            cap.kind(),
            cap.target()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn everything_is_denied_with_reason() {
        let broker = PermissionBroker::new(Permissions {
            fs: vec!["read:workspace".to_string()],
            ..Default::default()
        });
        for cap in [
            Capability::FsRead("/etc/passwd".to_string()),
            Capability::FsWrite("/tmp/x".to_string()),
            Capability::Net("localhost".to_string()),
            Capability::Exec("pi".to_string()),
        ] {
            let d = broker.check(&cap);
            assert!(!d.is_granted());
            match d {
                Decision::Denied(reason) => assert!(reason.contains("denied")),
                Decision::Granted => panic!("v1 must never grant"),
            }
        }
        // Requested perms are retained for display even though nothing is granted.
        assert_eq!(broker.requested().fs, vec!["read:workspace"]);
    }
}
