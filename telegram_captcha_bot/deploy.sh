#!/bin/bash
# Скрипт деплоя Telegram-бота на Contabo

set -e

echo "=== Деплой Telegram Captcha Bot на Contabo ==="

# Конфигурация
SERVER="yehhi@13.140.153.27"
REMOTE_DIR="~/captcha_bot"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "1. Загрузка файлов на сервер..."
ssh $SERVER "mkdir -p $REMOTE_DIR"
scp -r "$LOCAL_DIR"/* $SERVER:$REMOTE_DIR/

echo "2. Подключение к серверу..."
ssh $SERVER << 'ENDSSH'
cd ~/captcha_bot

echo "3. Создание виртуального окружения..."
python3 -m venv venv
source venv/bin/activate

echo "4. Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo "5. Создание .env файла..."
if [ ! -f .env ]; then
    cat > .env << 'EOL'
BOT_TOKEN=9287950312:AAHev0jcnuHs76tpBK87dGG9CiZ2-rOz_Lg
ADMIN_ID=8251441645
EOL
    echo "✅ .env создан"
else
    echo "⚠️  .env уже существует, пропускаем"
fi

echo "6. Остановка старого бота (если есть)..."
pm2 stop captcha_bot 2>/dev/null || true
pm2 delete captcha_bot 2>/dev/null || true

echo "7. Запуск бота через PM2..."
pm2 start bot.py --name captcha_bot --interpreter python3
pm2 save

echo "8. Настройка автозапуска..."
pm2 startup systemd -u yehhi --hp /home/yehhi

echo ""
echo "=== Деплой завершён! ==="
echo ""
echo "Проверка статуса:"
pm2 status
echo ""
echo "Просмотр логов:"
pm2 logs captcha_bot
echo ""
echo "Бот работает! Откройте @gesturelab_contact_bot в Telegram"
ENDSSH

echo ""
echo "✅ Готово! Бот развёрнут на Contabo"
