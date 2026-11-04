"""
Telegram-бот с капчей для защиты от спама.

Новые пользователи должны пройти капчу (кнопка + математическая задача),
прежде чем их сообщения пересылаются админу.
"""
import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Dict, Set

from dotenv import load_dotenv

# Загружаем .env файл
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

# Конфигурация
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))  # Ваш Telegram user ID

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не установлен. Создайте .env файл.")
if ADMIN_ID == 0:
    raise ValueError("ADMIN_ID не установлен. Укажите ваш Telegram user ID.")

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Состояния
class CaptchaStates(StatesGroup):
    WAITING_CAPTCHA = State()

# Хранилище проверенных пользователей
verified_users: Set[int] = set()

# Генерация капчи
def generate_captcha() -> tuple[str, int]:
    """Генерирует математическую задачу и ответ."""
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    question = f"{a} + {b} = ?"
    answer = a + b
    return question, answer

# Клавиатуры
def get_captcha_keyboard(question: str) -> InlineKeyboardMarkup:
    """Клавиатура с капчей."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Я не робот", callback_data="captcha_not_robot")],
        [InlineKeyboardButton(text=f"🧮 Решите: {question}", callback_data="captcha_math")],
    ])

def get_math_keyboard(answer: int) -> InlineKeyboardMarkup:
    """Клавиатура с вариантами ответа."""
    # Генерируем варианты (правильный + 3 неправильных)
    options = {answer}
    while len(options) < 4:
        wrong = answer + random.randint(-3, 3)
        if wrong > 0 and wrong != answer:
            options.add(wrong)
    
    options_list = list(options)
    random.shuffle(options_list)
    
    keyboard = []
    row = []
    for i, opt in enumerate(options_list):
        row.append(InlineKeyboardButton(
            text=str(opt),
            callback_data=f"captcha_answer_{opt}_{answer}"
        ))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Роутер
router = Router()

@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """Обработка /start."""
    user_id = message.from_user.id
    
    if user_id == ADMIN_ID:
        await message.answer("👋 Привет, админ! Бот работает.")
        return
    
    if user_id in verified_users:
        await message.answer("✅ Вы уже проверены. Можете писать сообщения.")
        return
    
    # Отправляем капчу
    question, answer = generate_captcha()
    await state.update_data(captcha_answer=answer)
    await state.set_state(CaptchaStates.WAITING_CAPTCHA)
    
    await message.answer(
        "👋 Привет! Чтобы защитить админа от спама, нужно пройти простую проверку.\n\n"
        "Нажмите кнопку «Я не робот», затем решите математическую задачу:",
        reply_markup=get_captcha_keyboard(question)
    )

@router.callback_query(F.data == "captcha_not_robot")
async def captcha_not_robot(callback: CallbackQuery, state: FSMContext):
    """Первый шаг капчи — кнопка «Я не робот»."""
    user_id = callback.from_user.id
    data = await state.get_data()
    answer = data.get("captcha_answer")
    
    if not answer:
        await callback.answer("❌ Капча устарела. Начните заново: /start")
        return
    
    await callback.answer("✅ Отлично! Теперь решите пример:")
    await callback.message.edit_text(
        f"🧮 Решите пример, чтобы подтвердить, что вы не робот:",
        reply_markup=get_math_keyboard(answer)
    )

@router.callback_query(F.data.startswith("captcha_answer_"))
async def captcha_answer(callback: CallbackQuery, state: FSMContext):
    """Проверка ответа на математическую задачу."""
    user_id = callback.from_user.id
    
    # Парсим callback_data: captcha_answer_{user_answer}_{correct_answer}
    parts = callback.data.split("_")
    user_answer = int(parts[2])
    correct_answer = int(parts[3])
    
    if user_answer == correct_answer:
        # Капча пройдена
        verified_users.add(user_id)
        await state.clear()
        
        await callback.answer("✅ Капча пройдена!")
        await callback.message.edit_text(
            "✅ Отлично! Вы прошли проверку.\n\n"
            "Теперь можете писать сообщения — они будут пересланы админу."
        )
        
        # Уведомляем админа
        try:
            await callback.bot.send_message(
                ADMIN_ID,
                f"👤 Новый пользователь: @{callback.from_user.username or 'без username'} "
                f"({callback.from_user.first_name}) прошёл капчу."
            )
        except Exception as e:
            logger.error(f"Не удалось уведомить админа: {e}")
    else:
        # Неверный ответ
        await callback.answer("❌ Неверно! Попробуйте ещё раз: /start")
        await state.clear()
        await callback.message.edit_text(
            "❌ Неверный ответ. Начните заново: /start"
        )

@router.message(CaptchaStates.WAITING_CAPTCHA)
async def waiting_captcha(message: Message):
    """Если пользователь пишет до прохождения капчи."""
    await message.answer("⚠️ Сначала пройдите капчу: /start")

@router.message(F.text)
async def forward_to_admin(message: Message):
    """Пересылка сообщений от проверенных пользователей админу."""
    user_id = message.from_user.id
    
    if user_id == ADMIN_ID:
        return
    
    if user_id not in verified_users:
        await message.answer("⚠️ Сначала пройдите капчу: /start")
        return
    
    # Пересылаем сообщение админу
    try:
        await message.copy_to(
            ADMIN_ID,
            caption=f"📨 От: @{message.from_user.username or 'без username'} "
                   f"({message.from_user.first_name})"
        )
        await message.answer("✅ Сообщение отправлено админу.")
    except Exception as e:
        logger.error(f"Не удалось переслать сообщение: {e}")
        await message.answer("❌ Ошибка при отправке. Попробуйте позже.")

@router.message()
async def handle_other(message: Message):
    """Обработка остальных типов сообщений (фото, видео и т.д.)."""
    user_id = message.from_user.id
    
    if user_id == ADMIN_ID:
        return
    
    if user_id not in verified_users:
        await message.answer("⚠️ Сначала пройдите капчу: /start")
        return
    
    # Пересылаем медиа админу
    try:
        await message.copy_to(
            ADMIN_ID,
            caption=f"📨 От: @{message.from_user.username or 'без username'} "
                   f"({message.from_user.first_name})"
        )
        await message.answer("✅ Сообщение отправлено админу.")
    except Exception as e:
        logger.error(f"Не удалось переслать сообщение: {e}")
        await message.answer("❌ Ошибка при отправке. Попробуйте позже.")

async def main():
    """Запуск бота."""
    bot = Bot(token=BOT_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(router)
    
    logger.info("Бот запускается...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
