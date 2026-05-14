"""Decision-point modal — Accept / Override / Skip triad.

Shown after natural pipeline breaks (post-discover DAG select, post-identify
non-identifiable override, post-estimate prefer change, etc.). The choice is
recorded on the StudyProtocol's ``decision_ledger`` so the report can show
who decided what and why.
"""

from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class DecisionModal(ModalScreen[str]):
    """A small, dim-veiled modal with three buttons.

    ``await app.push_screen_wait(DecisionModal(...))`` returns one of
    ``"accept" | "override" | "skip"``.
    """

    DEFAULT_CSS = ""

    BINDINGS = [
        ("escape", "skip", "Skip"),
        ("enter", "accept", "Accept"),
    ]

    class Choice(Message):
        def __init__(self, choice: str) -> None:
            self.choice = choice
            super().__init__()

    def __init__(
        self,
        title: str,
        prose: str,
        accept_label: str = "Accept default",
        override_label: str = "Override",
        skip_label: str = "Skip",
    ) -> None:
        super().__init__()
        self._title = title
        self._prose = prose
        self._accept_label = accept_label
        self._override_label = override_label
        self._skip_label = skip_label

    def compose(self):
        with Vertical():
            yield Static(self._title, classes="modal-title")
            yield Static(self._prose, classes="modal-prose")
            with Horizontal():
                yield Button(self._accept_label, id="accept", classes="primary")
                yield Button(self._override_label, id="override")
                yield Button(self._skip_label, id="skip")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id is not None:
            self.dismiss(event.button.id)

    def action_accept(self) -> None:
        self.dismiss("accept")

    def action_skip(self) -> None:
        self.dismiss("skip")
