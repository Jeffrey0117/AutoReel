"""
Telegram Bot Service (stub) - Telegram bot integration.
"""


class TelegramBotService:
    def __init__(self):
        self._is_running = False
        self.notifier = None

    def set_event_loop(self, loop):
        pass

    def set_ws_manager(self, manager):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass


telegram_bot_service = TelegramBotService()
