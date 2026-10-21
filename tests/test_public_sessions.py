from __future__ import annotations

import unittest

from pde_agent_verifier_solver.sdk import (
    Case,
    CaseSource,
    CaseVisibility,
    open_public_case_session,
)
from pde_agent_verifier_solver.sdk.public_sessions import (
    bind_public_session_factory,
)


class _AdversarialSessionTarget:
    def __init__(self) -> None:
        self.secret_value = "hidden"
        self.close_calls = 0

    def query_public(self, value: int) -> int:
        return value + 1

    def query_secret(self) -> str:
        return self.secret_value

    def __getitem__(self, key: str) -> str:
        return f"hidden:{key}"

    def close(self) -> None:
        self.close_calls += 1


class PublicSessionAllowlistTests(unittest.TestCase):
    def test_proxy_exposes_only_allowlisted_methods_and_revokes_them(self) -> None:
        case = Case(
            case_id="allowlist-case",
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN,
            public_data={"operator": "test"},
        )
        target = _AdversarialSessionTarget()

        with bind_public_session_factory(
            lambda opened_case: target,
            case,
            allowed_methods=("query_public",),
        ):
            session = open_public_case_session(case)
            method = session.query_public
            self.assertEqual(method(3), 4)

            for name in (
                "_target",
                "_binding",
                "secret_value",
                "query_secret",
            ):
                with self.subTest(name=name):
                    with self.assertRaises(AttributeError):
                        getattr(session, name)
            with self.assertRaises(TypeError):
                session["secret"]

        self.assertEqual(target.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            method(4)
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            session.query_public(4)


if __name__ == "__main__":
    unittest.main()
