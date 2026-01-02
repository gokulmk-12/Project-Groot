import reflex as rx
from app.components import navbar, chat

def index() -> rx.Component:
    return rx.vstack(
            navbar.navbar(),
            chat.chat(),
            chat.action_bar(),
            height="100dvh",
            align_items="stretch",
            spacing="0",
        )


app = rx.App(
    theme=rx.theme(
        appearance="inherit",
        accent_color="orange",
    )
)
app.add_page(index)
