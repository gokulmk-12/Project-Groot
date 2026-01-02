import reflex as rx
from app.state import State, QA
from reflex.constants.colors import ColorType


def message_content(text: str, color: ColorType) -> rx.Component:
    return rx.markdown(
        text,
        max_width="100%",
        overflow_wrap='anywhere',
        background_color=rx.color(color, 4),
        color=rx.color(color, 12),
        display="inline-block",
        padding_inline="1em",
        border_radius="12px",
    )

def message(qa: QA) -> rx.Component:
    return rx.box(
        rx.box(
            message_content(qa["question"], "mauve"), 
            text_align="right",
            margin_top="40px",
            margin_bottom="10px"
        ),
        rx.box(
            message_content(qa["answer"], "accent"), 
            text_align="left",
            margin_bottom="8px"
        ),
        max_width="50em",
        margin_inline="auto",
    )

def chat() -> rx.Component:
    return rx.auto_scroll(
        rx.foreach(State.selected_chat, message),
        flex="1",
        padding="20px",
    )

def action_bar() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.form(
                rx.center(
                    rx.box(
                        rx.hstack(
                            rx.input(
                                placeholder="Ask anything",
                                id="question",
                                flex="1",
                                variant="soft",
                                size="3",
                                border_radius="999px",
                                padding_left="16px",
                                padding_right="12px",
                                background_color=rx.color("mauve", 1),
                                auto_complete=False,
                            ),
                            rx.icon_button(
                                rx.icon("arrow_up", size=20),
                                loading=State.processing,
                                disabled=State.processing,
                                type="submit",
                                border_radius="999px",
                                height="40px",
                                width="40px",
                                variant="solid",
                                color_scheme="orange",
                            ),
                            align_items="center",
                            spacing="2",
                        ),
                        max_width="42em",
                        width="100%",
                        padding="6px",
                        border_radius="999px",
                        border=f"1px solid {rx.color('mauve', 4)}",
                        background_color=rx.color("mauve", 2),
                        box_shadow="0 8px 24px rgba(0,0,0,0.06)",
                    ),
                    width="100%",
                ),
                reset_on_submit=True,
                on_submit=State.process_question,
            ),
            rx.text(
                "Groot may return factually incorrect or misleading responses.",
                text_align="center",
                font_size=".75em",
                color=rx.color("mauve", 10),
            ),
            width="100%",
            padding_x="16px",
            align="stretch",
        ),
        position="sticky",
        bottom="0",
        width="100%",
        padding_y="16px",
        backdrop_filter="blur(12px)",
        border_top=f"1px solid {rx.color('mauve', 3)}",
    )