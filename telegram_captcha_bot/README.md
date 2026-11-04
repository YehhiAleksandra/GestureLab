# Telegram Captcha Bot

Бот с капчей для защиты от спама. Новые пользователи должны пройти проверку (кнопка + математическая задача), прежде чем их сообщения пересылаются админу.

## Возможности

- ✅ Капча: кнопка "Я не робот" + математическая задача
- ✅ Пересылка сообщений админу после проверки
- ✅ Защита от спама
- ✅ Простая настройка

## Установка

### 1. Создайте бота

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Придумайте имя и username
4. Скопируйте токен

### 2. Узнайте свой User ID

1. Откройте [@userinfobot](https://t.me/userinfobot) в Telegram
2. Отправьте `/start`
3. Скопируйте ваш ID

### 3. Настройте окружение

```bash
cd telegram_captcha_bot
cp .env.example .env
nano .env  # или другой редактор
```

Заполните:
```
BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
ADMIN_ID=123456789
```

### 4. Установите зависимости

```bash
pip install -r requirements.txt
```

### 5. Запустите бота

```bash
python bot.py
```

## Деплой на Contabo

### 1. Загрузите на сервер

```bash
scp -r telegram_captcha_bot yehhi@13.140.153.27:~/captcha_bot
```

### 2. Настройте на сервере

```bash
ssh yehhi@13.140.153.27
cd ~/captcha_bot

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt

# Настройте .env
cp .env.example .env
nano .env
```

### 3. Запустите через PM2

```bash
# Установите PM2 (если не установлен)
npm install -g pm2

# Запустите бота
pm2 start bot.py --name captcha_bot --interpreter python3

# Сохраните конфигурацию
pm2 save
pm2 startup
```

### 4. Проверьте

```bash
pm2 status
pm2 logs captcha_bot
```

## Использование

1. Откройте бота в Telegram
2. Отправьте `/start`
3. Пройдите капчу (кнопка + математическая задача)
4. Пишите сообщения — они пересылаются админу

## Структура

```
telegram_captcha_bot/
├── bot.py              # Основной код бота
├── requirements.txt    # Зависимости Python
├── .env.example       # Пример конфигурации
└── README.md          # Документация
```

## Безопасность

- Токен бота хранится в `.env` (не коммитить!)
- `.env` добавлен в `.gitignore`
- Админ ID захардкожен в `.env`

## Лицензия

MIT
