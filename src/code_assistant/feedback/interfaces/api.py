
from code_assistant.feedback.interface import FeedbackInterface
from code_assistant.feedback.models import FeedbackRequest


class APIFeedbackInterface(FeedbackInterface):

    def request_feedback(self, request: FeedbackRequest) -> str:
        print(request)
        return "Feedback requested"
