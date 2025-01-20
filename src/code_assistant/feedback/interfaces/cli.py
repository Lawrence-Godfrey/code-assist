from rich.console import Console
from rich.markdown import Markdown

from code_assistant.feedback.interface import FeedbackInterface
from code_assistant.feedback.models import FeedbackRequest
from code_assistant.feedback.exceptions import FeedbackCancelled

class CLIFeedbackInterface(FeedbackInterface):
    """Command-line interface for collecting feedback."""

    def __init__(self):
        self._console = Console()

    def request_feedback(self, request: FeedbackRequest) -> str:
        """
        Request feedback via command line interface.

        Displays the prompt and collects input from the user.

        Args:
            request: The feedback request.

        Returns:
            The user's response as a string.

        Raises:
            FeedbackCancelled: If user enters 'q' or 'quit'
        """
        # Display the prompt
        self._console.print("\n" + "=" * 80)
        self._console.print("[bold blue]Feedback Required[/bold blue]")
        self._console.print(Markdown(request.prompt))
        self._console.print(
            "\n[dim](Enter 'q' or 'quit' to cancel at any time)[/dim]"
        )

        # Get user input
        response = input("\nYour response: ").strip()

        # Check for quit command
        if response.lower() in ('q', 'quit'):
            raise FeedbackCancelled("Feedback collection cancelled by user")

        return response