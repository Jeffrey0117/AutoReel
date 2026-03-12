"""
Telegram Notifications (stub) - Send notifications via Telegram.
"""


class TelegramNotifier:
    def __init__(self, bot_service):
        self.bot_service = bot_service

    async def on_broadcast(self, message):
        pass
