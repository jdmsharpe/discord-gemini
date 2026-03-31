from discord_gemini import Conversation


class TestConversation:
    def test_conversation_dataclass(self):
        """Test the Conversation dataclass."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        history = [{"role": "user", "parts": [{"text": "Hello"}]}]
        conversation = Conversation(params=params, history=history)

        assert conversation.params == params
        assert conversation.history == history
