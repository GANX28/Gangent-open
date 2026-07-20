import unittest

from gangent.secret_guard import (
    contains_secret,
    is_sensitive_path,
    redact_data,
    redact_secrets,
    scan_secrets,
)


class SecretGuardTests(unittest.TestCase):
    def test_sensitive_path_detection(self):
        self.assertTrue(is_sensitive_path(".env"))
        self.assertTrue(is_sensitive_path(".ssh/id_rsa"))
        self.assertTrue(is_sensitive_path("private.pem"))
        self.assertFalse(is_sensitive_path("gangent/secret_guard.py"))

    def test_secret_scanning_and_redaction(self):
        text = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456"

        findings = scan_secrets(text)
        redacted = redact_secrets(text)

        self.assertTrue(findings)
        self.assertTrue(contains_secret(text))
        self.assertIn("[REDACTED_", redacted)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", redacted)

    def test_redact_data_recursively(self):
        data = {
            "message": "token=abcdefgh12345678",
            "items": ["password=abcdefghijkl"],
        }

        redacted = redact_data(data)

        self.assertIn("[REDACTED_", redacted["message"])
        self.assertIn("[REDACTED_", redacted["items"][0])


if __name__ == "__main__":
    unittest.main()
