//! umber-update — optional signed self-updater.
//!
//! Exposed as the toggleable module `updater` (kernel `FeatureRegistry`) backed
//! by the `auto_update` config boolean. When disabled it never runs. Every
//! network/verify/IO error is **non-fatal and fail-closed**: the caller aborts
//! the swap and launches on the current binary regardless. A binary is only
//! swapped in after its detached ed25519 signature verifies against the
//! embedded release public key.

use std::error::Error;
use std::io::{Read, Write};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

pub const RELEASE_OWNER: &str = "Myers-Technologies-Public";
pub const RELEASE_REPO: &str = "umber";

/// ed25519 release public key (64 hex chars) matching the CI signing key
/// (`UMBER_SIGN_KEY`). Empty until the keypair is generated — while empty,
/// verification refuses and the updater self-disables (fail closed).
pub const RELEASE_PUBKEY_HEX: &str = "";

type Res<T> = Result<T, Box<dyn Error>>;

/// Result of an update check.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Already on the latest (or a newer) build.
    UpToDate,
    /// A newer signed build was verified and swapped in; the caller should
    /// re-exec the (now replaced) current executable.
    Updated { version: String },
}

/// True if `latest` is strictly newer than `current` (`vX.Y.Z` or `X.Y.Z`).
/// Unparseable components sort as 0.
pub fn is_newer(latest: &str, current: &str) -> bool {
    parse(latest) > parse(current)
}

fn parse(v: &str) -> (u64, u64, u64) {
    let v = v.trim().trim_start_matches('v');
    let mut it = v.split(['.', '-', '+']);
    let n = |s: Option<&str>| s.and_then(|s| s.parse().ok()).unwrap_or(0);
    (n(it.next()), n(it.next()), n(it.next()))
}

/// Platform raw-binary asset name + its detached-signature name, matching the
/// artifacts produced by `.github/workflows/release.yml`.
pub fn asset_names() -> (&'static str, &'static str) {
    #[cfg(windows)]
    {
        ("umber-x86_64-windows.exe", "umber-x86_64-windows.exe.sig")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        ("umber-x86_64-linux", "umber-x86_64-linux.sig")
    }
    #[cfg(target_os = "macos")]
    {
        ("umber-x86_64-macos", "umber-x86_64-macos.sig")
    }
}

fn http_get_string(url: &str) -> Res<String> {
    Ok(ureq::get(url)
        .set("User-Agent", "umber-update")
        .call()?
        .into_string()?)
}

fn http_get_bytes(url: &str) -> Res<Vec<u8>> {
    let resp = ureq::get(url).set("User-Agent", "umber-update").call()?;
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn hex32(h: &str) -> Res<[u8; 32]> {
    if h.len() != 64 {
        return Err("release public key must be 64 hex chars".into());
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16)?;
    }
    Ok(out)
}

/// Verify `sig` (64 raw bytes) over `bin` against the embedded release key.
fn verify(bin: &[u8], sig: &[u8]) -> Res<()> {
    let vk = VerifyingKey::from_bytes(&hex32(RELEASE_PUBKEY_HEX)?)?;
    let raw: [u8; 64] = sig.try_into().map_err(|_| "signature must be 64 bytes")?;
    vk.verify(bin, &Signature::from_bytes(&raw))?;
    Ok(())
}

/// Check the latest GitHub release; if newer than `current_version`, download
/// the platform binary + signature, verify ed25519, and self-replace the
/// running executable. Fail-closed: any error aborts the swap.
pub fn check_and_apply(current_version: &str) -> Res<Outcome> {
    let api =
        format!("https://api.github.com/repos/{RELEASE_OWNER}/{RELEASE_REPO}/releases/latest");
    let rel: serde_json::Value = serde_json::from_str(&http_get_string(&api)?)?;
    let tag = rel["tag_name"].as_str().ok_or("release has no tag_name")?;
    if !is_newer(tag, current_version) {
        return Ok(Outcome::UpToDate);
    }

    let (bin_name, sig_name) = asset_names();
    let assets = rel["assets"].as_array().ok_or("release has no assets")?;
    let url = |name: &str| -> Option<String> {
        assets.iter().find_map(|a| {
            (a["name"].as_str() == Some(name))
                .then(|| a["browser_download_url"].as_str().map(str::to_owned))
                .flatten()
        })
    };
    let bin_url = url(bin_name).ok_or("no binary asset for this platform")?;
    let sig_url = url(sig_name).ok_or("no signature asset for this platform")?;

    let bin = http_get_bytes(&bin_url)?;
    let sig = http_get_bytes(&sig_url)?;
    verify(&bin, &sig)?;

    // Stage beside the current exe, mark executable, then swap atomically.
    let exe = std::env::current_exe()?;
    let dir = exe.parent().ok_or("current exe has no parent directory")?;
    let tmp = dir.join(format!(".umber-update-{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bin)?;
        f.flush()?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    if let Err(e) = self_replace::self_replace(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    let _ = std::fs::remove_file(&tmp);
    Ok(Outcome::Updated {
        version: tag.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_ordering() {
        assert!(is_newer("v0.2.0", "v0.1.9"));
        assert!(is_newer("0.1.10", "0.1.9")); // numeric, not lexical
        assert!(is_newer("v1.0.0", "0.9.9"));
        assert!(!is_newer("v0.1.0", "v0.1.0"));
        assert!(!is_newer("v0.1.0", "v0.2.0"));
    }

    #[test]
    fn verify_fails_closed_without_key() {
        // RELEASE_PUBKEY_HEX is empty until the keypair is generated.
        assert!(verify(b"payload", &[0u8; 64]).is_err());
    }
}
