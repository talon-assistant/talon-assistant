"""Centralized credential storage using the OS keyring.

Wraps the `keyring` library to provide a single interface for all talents
to store and retrieve secrets (API keys, passwords, tokens).  Falls back
to plaintext config with a console warning when keyring is unavailable.

Key format:  service = "talon_assistant"
             username = "{talent_name}.{field_key}"
"""

try:
    import keyring
    _HAS_KEYRING = True
except ImportError:
    _HAS_KEYRING = False

_SERVICE = "talon_assistant"


class CredentialStore:
    """OS-level secret storage for Talon talent credentials."""

    # Legacy service names from before centralisation (email_talent used this)
    _LEGACY_SERVICES = {"talon_email"}

    def __init__(self):
        if not _HAS_KEYRING:
            print("   [CredentialStore] WARNING: keyring not installed. "
                  "Secrets will remain in plaintext config.")

    # ── Public API ────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when the keyring backend is usable."""
        return _HAS_KEYRING

    def store_secret(self, talent_name: str, field_key: str, value: str) -> bool:
        """Store a secret in the OS keyring.

        Returns True on success, False if keyring is unavailable or errored.
        """
        if not _HAS_KEYRING or not value:
            return False
        try:
            keyring.set_password(_SERVICE, f"{talent_name}.{field_key}", value)
            return True
        except Exception as e:
            print(f"   [CredentialStore] Failed to store "
                  f"{talent_name}.{field_key}: {e}")
            return False

    def get_secret(self, talent_name: str, field_key: str) -> str:
        """Retrieve a secret from the OS keyring.

        Returns the secret string, or "" if not found / unavailable.
        """
        if not _HAS_KEYRING:
            return ""
        try:
            value = keyring.get_password(_SERVICE, f"{talent_name}.{field_key}")
            return value or ""
        except Exception as e:
            print(f"   [CredentialStore] Failed to read "
                  f"{talent_name}.{field_key}: {e}")
            return ""

    def delete_secret(self, talent_name: str, field_key: str) -> bool:
        """Remove a secret from the OS keyring."""
        if not _HAS_KEYRING:
            return False
        try:
            keyring.delete_password(_SERVICE, f"{talent_name}.{field_key}")
            return True
        except keyring.errors.PasswordDeleteError:
            return False  # already absent
        except Exception as e:
            print(f"   [CredentialStore] Failed to delete "
                  f"{talent_name}.{field_key}: {e}")
            return False

    def has_secret(self, talent_name: str, field_key: str) -> bool:
        """Check whether a secret exists without returning its value."""
        return bool(self.get_secret(talent_name, field_key))

    # ── Legacy migration ──────────────────────────────────────────

    def migrate_legacy_email(self, username: str) -> None:
        """Migrate email password from old 'talon_email' service to the
        new centralised 'talon_assistant' service.

        Called once during startup.  Safe to call repeatedly (no-op if
        already migrated or no legacy entry exists).
        """
        if not _HAS_KEYRING or not username:
            return
        try:
            old_pw = keyring.get_password("talon_email", username)
            if old_pw:
                self.store_secret("email", "password", old_pw)
                try:
                    keyring.delete_password("talon_email", username)
                except Exception:
                    pass  # non-critical
                print("   [CredentialStore] Migrated email password "
                      "from legacy keyring entry")
        except Exception:
            pass  # no legacy entry — nothing to do
