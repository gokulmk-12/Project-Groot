import reflex as rx
import asyncio
import requests
from typing import Any, TypedDict

class QA(TypedDict):
    question: str
    answer: str

class State(rx.State):
    _chats: dict[str, list[QA]] = {
        "Intros": [],
    }

    current_chat = "Intros"
    processing: bool = False
    is_modal_open: bool = False
    
    @rx.event
    def create_chat(self, form_data: dict[str, Any]):
        new_chat_name = form_data["new_chat_name"]
        self.current_chat = new_chat_name
        self._chats[new_chat_name] = []
        self.is_modal_open = False
    
    @rx.event
    def set_is_modal_open(self, is_open: bool):
        self.is_modal_open = is_open
    
    @rx.var
    def selected_chat(self) -> list[QA]:
        return (
            self._chats[self.current_chat] if self.current_chat in self._chats else []
        )
    
    @rx.event
    def delete_chat(self, chat_name: str):
        if chat_name not in self._chats:
            return
        del self._chats[chat_name]
        if len(self._chats) == 0:
            self._chats = {
                "Intros": [],
            }
        if self.current_chat not in self._chats:
            self.current_chat = list(self._chats.keys())[0]
    
    @rx.event
    def set_chat(self, chat_name: str):
        self.current_chat = chat_name
    
    @rx.event
    def set_new_chat_name(self, new_chat_name: str):
        self.new_chat_name = new_chat_name
    
    @rx.var
    def chat_titles(self) -> list[str]:
        return list(self._chats.keys())

    @rx.event
    async def process_question(self, form_data: dict[str, Any]):
        question = form_data["question"]
        if not question:
            return
        
        async for value in self.process_answer(question):
            yield value
        
    @rx.event
    async def process_answer(self, question: str):
        qa = QA(question=question, answer="")
        self._chats[self.current_chat].append(qa)

        self.processing = True
        yield

        resp = await asyncio.to_thread(
            requests.post,
            "http://localhost:8080/generate",
            json={
                "prompt": question,
                "max_new_tokens": 200,
                "temperature": 0.8
            },
            timeout=60,
        )
        answer = resp.json()["content"]

        for i in range(len(answer)):
            await asyncio.sleep(0.01)
            self._chats[self.current_chat][-1]["answer"] = answer[:i+1]
            yield
        self.processing = False
