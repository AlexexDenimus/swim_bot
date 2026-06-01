import asyncio
from os import getenv

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv

from swimbot.config import TMP_DIR
from swimbot_data.references import REFERENCE_VIDEOS
from swimbot_models.move_tracker import compare_videos, extract_keypoints

load_dotenv()
TOKEN = getenv("BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

REFERENCE_KEYPOINTS: dict[str, dict] = {}
USER_STYLE: dict[int, str] = {}

_ANALYSIS_SEM = asyncio.Semaphore(1)


def get_inline_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Брасс", callback_data="bras")],
            [InlineKeyboardButton(text="Кроль", callback_data="crawl")],
        ],
    )
    return keyboard


async def warmup_references():
    loop = asyncio.get_running_loop()
    for key, path in REFERENCE_VIDEOS.items():
        print(f"Extracting keypoints for {key}...", flush=True)
        REFERENCE_KEYPOINTS[key] = await loop.run_in_executor(
            None,
            lambda path=path: extract_keypoints(
                str(path),
                False,
                use_orientation_flip=False,
                preprocess=False,
                frame_stride=2,
            ),
        )
    print("Reference warmup complete.", flush=True)


async def send_reference_video(callback: CallbackQuery, video_key: str):
    USER_STYLE[callback.from_user.id] = video_key

    video_path = REFERENCE_VIDEOS[video_key]
    if not video_path.exists():
        await callback.message.answer(f"Файл с примером не найден: {video_path}")
        return

    await callback.message.answer("Посмотрите пример и отправьте видео")
    await callback.message.answer_video(video=FSInputFile(video_path))


@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer(
        "Выберите стиль для плавания", reply_markup=get_inline_keyboard()
    )


@dp.message(F.video)
async def video_message(message: Message):
    user_id = message.from_user.id
    style_key = USER_STYLE.get(user_id)

    if style_key is None:
        await message.answer(
            "Сначала выберите стиль плавания", reply_markup=get_inline_keyboard()
        )
        return

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_in = TMP_DIR / f"{user_id}_in.mp4"
    tmp_out = TMP_DIR / f"{user_id}_out.mp4"

    await bot.download(message.video, destination=str(tmp_in))
    await message.answer("Анализирую видео, это займёт ~30-60 секунд...")
    await bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_VIDEO)

    try:
        async with _ANALYSIS_SEM:
            result = await asyncio.to_thread(
                compare_videos,
                str(tmp_in),
                REFERENCE_KEYPOINTS[style_key],
                str(tmp_out),
                user_kwargs={
                    "use_orientation_flip": False,
                    "preprocess": False,
                    "frame_stride": 2,
                },
                output_max_side=720,
            )

        sim_pct = result["weighted_similarity"] * 100
        if sim_pct >= 70:
            verdict = "Отлично!"
        elif sim_pct >= 50:
            verdict = "Неплохо, но есть над чем поработать."
        else:
            verdict = "Нужно больше практики."

        user_fps = result.get("user_fps", 30) or 30
        j_start = result.get("j_start", 0)
        j_end = result.get("j_end", 0)
        user_n_frames = result.get("user_n_frames", j_end + 1)

        trim_parts = []
        if j_start > 0:
            trim_parts.append(f"начало {j_start / user_fps:.1f}с")
        trailing = user_n_frames - 1 - j_end
        if trailing > 0:
            trim_parts.append(f"конец {trailing / user_fps:.1f}с")

        await message.answer_video(
            video=FSInputFile(str(tmp_out)),
            caption=f"{verdict}\nСходство: {sim_pct:.1f}%",
            supports_streaming=True,
        )
    except Exception as exc:
        await message.answer(f"Ошибка при анализе: {exc}")
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


@dp.callback_query(F.data == "bras")
async def bras_callback(callback: CallbackQuery):
    await send_reference_video(callback, "bras")


@dp.callback_query(F.data == "crawl")
async def crawl_callback(callback: CallbackQuery):
    await send_reference_video(callback, "crawl")


async def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print("Bot is running...")
    await warmup_references()
    await dp.start_polling(bot)
