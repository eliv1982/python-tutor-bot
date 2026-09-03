# Personal Python Tutor — Telegram Bot

Мультимодальный бот-тьютор по Python: текст, голос, изображения, RAG (база знаний), генерация картинок (DALL-E 3).

Репозиторий: [github.com/eliv1982/python-tutor-bot-](https://github.com/eliv1982/python-tutor-bot-)

---

## Запуск

1. Скопируйте `.env.example` в `.env` и заполните:
   - `TELEGRAM_BOT_TOKEN` — токен от [@BotFather](https://t.me/BotFather)
   - `OPENAI_API_KEY` — ключ OpenAI (обязателен всегда: embeddings, Vision, STT, TTS, генерация изображений)
   - `LLM_PROVIDER` — провайдер основного текстового диалога/RAG-ответов: `anthropic` (по умолчанию) или `openai`
   - `ANTHROPIC_API_KEY` — ключ Anthropic (обязателен, если `LLM_PROVIDER=anthropic`)
   - `TELEGRAM_ALLOWED_USER_IDS` — числовые Telegram user id, кому разрешён доступ (через запятую)
   - `DATABASE_URL` — строка подключения к PostgreSQL (см. раздел «PostgreSQL» ниже)
2. Установите зависимости: `pip install -r requirements.txt`
3. Поднимите PostgreSQL (локально или в контейнере) и примените миграции: `alembic upgrade head`
4. Запуск: `python main.py`

---

## Доступ к боту и идентификация пользователей

Доступ к боту временно ограничен списком числовых Telegram user id в
`TELEGRAM_ALLOWED_USER_IDS` (`.env`). Если переменная не задана, пустая
или не содержит ни одного корректного id — бот отклоняет всех пользователей
(fail closed), а не становится публичным. Это остаётся временным механизмом
предварительной защиты до будущей системы OAuth-логина, а не окончательной
моделью авторизации приложения.

Начиная с этой стадии, Telegram-идентификатор — это **только внешняя
идентичность адаптера**. Сразу после прохождения проверки доступа
`from_user.id` разрешается в стабильный внутренний UUID (таблицы `users` +
`telegram_accounts` в PostgreSQL, см. `app/identity.py`/`db/identity.py`), и
именно этот UUID используется дальше как канонический идентификатор
пользователя — в состоянии диалога, владении документами и Qdrant. Создание
внутреннего пользователя происходит **только после** прохождения allowlist —
никогда как замена проверке доступа.

---

## Web-адаптер (Stage 6A)

Помимо Telegram-бота, репозиторий содержит отдельный FastAPI-адаптер
(`web/`) поверх того же канонического слоя идентичности — без второй
модели пользователя и без второго источника истины. Аутентификация
браузера — настоящая **серверная сессия** в PostgreSQL (не JWT, не
подписанный клиентский токен): cookie несёт только непрозрачный
криптографически случайный идентификатор, в базе хранится лишь его
SHA-256-дайджест (см. `db/auth_sessions.py`, `app/auth_session.py`).

- Запуск (отдельный процесс, не запускает Telegram-бота):
  ```
  python web_main.py
  ```
  Требует `SESSION_SECRET_KEY` в `.env` (подпись CSRF-токенов) —
  без него адаптер не запустится (fail closed), см. `.env.example`.
- Эндпоинты: `GET /healthz` (без авторизации), `GET /api/me` (текущий
  пользователь — только `id`/`created_at`, без Telegram id и внутренних
  деталей), `POST /api/logout` (требует валидную сессию и CSRF-заголовок
  `X-CSRF-Token`).
- CSRF: stateless double-submit cookie, значение криптографически привязано
  к самому токену сессии (`web/csrf.py`) — не единственная защита от CSRF,
  но SameSite=Lax остаётся дополнительным слоем, не основным механизмом.

### GitHub OAuth-логин (Stage 6B)

Браузер может дополнительно войти через настоящий OAuth 2.0 Authorization
Code + PKCE (S256) флоу GitHub — `GET /api/auth/github/login` (редирект на
GitHub) и `GET /api/auth/github/callback` (обмен кода, резолв личности,
выпуск сессии). GitHub здесь — **только внешний провайдер идентичности**:
канонический пользователь остаётся тем же `users.id` UUID, что и для
Telegram, а не второй моделью пользователя.

- Устойчивый внешний идентификатор — числовой `id` GitHub-аккаунта
  (`GET https://api.github.com/user`), НИКОГДА login/username (может
  измениться) и НИКОГДА email (может отсутствовать/быть приватным).
  Таблица `github_accounts` — привязка этого id к `users.id`, структурно
  идентична `telegram_accounts`.
- Access-токен GitHub — эфемерный bootstrap-креденшл: используется один раз
  сразу после обмена кода (запрос `GET /user`) и нигде не сохраняется, не
  логируется и не попадает в cookie/ответ. Область доступа (scope) не
  запрашивается вовсе — минимально необходимый для чтения публичного
  профиля уровень.
- OAuth-транзакция (`state` + PKCE `code_verifier`) хранится в PostgreSQL
  (`github_oauth_transactions`) короткоживущей (по умолчанию 10 минут,
  `GITHUB_OAUTH_TRANSACTION_TTL_SECONDS`) и одноразовой: атомарный
  `DELETE ... RETURNING` в `db/oauth_transactions.py` гарантирует, что
  повторное или параллельное использование одного и того же `state`
  успевает только один раз, а cleartext PKCE-verifier удаляется из базы
  сразу после использования (никогда не хранится бессрочно). Сохраняется
  лишь SHA-256 дайджест `state` (не само значение) — как и
  `session_token_hash` для web-сессий.
- Защита от login CSRF: `/login` дополнительно ставит короткоживущую
  HttpOnly-cookie с тем же `state`, что ушёл в GitHub; `/callback`
  требует точного совпадения query-параметра `state` с этой cookie ДО
  обращения к базе — иначе злоумышленник, легитимно начавший СВОЙ
  собственный вход, мог бы завлечь чужой браузер на callback-URL и
  подсадить жертве сессию от своего же GitHub-аккаунта. Cookie
  сбрасывается только когда query `state` совпал с ней (эта транзакция
  действительно текущая для браузера) — несовпадающий/старый callback
  никогда не сбрасывает cookie другой, всё ещё активной вкладки.
- После успешной идентификации всегда выпускается ЗАНОВО созданная сессия
  через `app/auth_session.create_session_for_github()` — переразрешает
  GitHub-id → канонический UUID заново, под блокировкой, в ТОЙ ЖЕ
  транзакции, что и вставка сессии, и отказывает (fail closed), если
  GitHub-маппинг к этому моменту вообще исчез (например, конкурентный
  unlink); сама эта функция `auth_generation` не проверяет и generation-
  aware не является — это отдельная, более узкая перепроверка. Проверку
  generation-staleness (Stage 6C corrective pass, independent-audit
  MAJOR 1: отклоняет попытку, если GitHub-маппинг был отвязан на более
  позднем `auth_generation`, чем тот, что зафиксирован при старте этого
  OAuth-флоу) выполняет ДО этого шага `app.github_identity.
  resolve_user_uuid_for_oauth()`, вызываемый в `web/github_oauth.py` перед
  `create_session_for_github()`. Здесь и далее "generation" — счётчик
  `auth_generation`/unlink race gate, а не лимит одновременных LLM text-
  generation запросов (см. `app/generation_limits.py`). Существующая
  сессия браузера (если была) не читается, не переиспользуется и не
  отзывается как побочный эффект чужого входа.
- Первый вход через GitHub **сам по себе** всегда создаёт свой отдельный
  канонический аккаунт — даже если тот же человек уже существует как
  Telegram-пользователь; автоматическое слияние по email/username/
  эвристике никогда не происходит. Явное связывание этих двух аккаунтов
  одним и тем же человеком — отдельный, сознательный шаг пользователя,
  см. "Связывание Telegram- и GitHub-аккаунтов (Stage 6C)" ниже.
- Требует `GITHUB_CLIENT_ID`/`GITHUB_CLIENT_SECRET`/`GITHUB_REDIRECT_URI` в
  `.env` (см. `.env.example`) — без них web-адаптер не запустится (fail
  closed), это обязательный маршрут адаптера, не опциональный флаг.
  `GITHUB_REDIRECT_URI` должен точно совпадать с callback URL,
  зарегистрированным в настройках GitHub OAuth App
  (https://github.com/settings/developers); используйте отдельные
  OAuth-приложения для разработки и продакшна. Допускается только
  https:// в продакшне, либо http:// на буквальном loopback-хосте
  (`127.0.0.1` или `::1` — НЕ `localhost`) при `WEB_ENV=development`; URI
  не может содержать userinfo/fragment/query, путь обязан быть ровно
  `/api/auth/github/callback`.

#### Ограничение хранилища OAuth-транзакций и глобальный rate limit (Stage 6B corrective pass #1)

Каждый `GET /api/auth/github/login` — неаутентифицированный маршрут.
Чтобы он не мог неограниченно расти в PostgreSQL, `db.oauth_transactions.
create_sync()` выполняет ВСЁ следующее одной атомарной транзакцией,
сериализованной между процессами через `SELECT ... FOR UPDATE` над
singleton-строкой `github_oauth_admission` (тот же приём, что
`web_session_policy` уже использует для Stage 6A cookie-posture):

1. удаляет все просроченные строки `github_oauth_transactions`
   (индекс `ix_github_oauth_transactions_expires_at`);
2. сбрасывает/продлевает фиксированное 60-секундное окно
   глобального rate limit (`GITHUB_OAUTH_MAX_STARTS_PER_MINUTE`,
   по умолчанию 30 стартов/минуту — глобально, не по IP: у приложения
   пока нет доверенного контракта на IP клиента за прокси);
3. отклоняет запрос (HTTP 429, строка НЕ создаётся) если окно rate limit
   исчерпано, ИЛИ если количество ещё живых транзакций уже достигло
   жёсткого потолка `GITHUB_OAUTH_MAX_OUTSTANDING_TRANSACTIONS`
   (по умолчанию 200) — оба предела configurable через `.env`, оба
   fail closed на некорректном/неположительном значении.

Это делает рост таблицы физически ограниченным даже без внешнего
cron/воркера и даже при нескольких процессах web-адаптера, делящих одну
БД. Внешний rate limit на reverse-proxy/ingress перед
`/api/auth/github/login` остаётся желательной defense-in-depth мерой
(см. ниже), но не единственной защитой хранилища.

#### Приватность callback (обязательное production-требование)

`GET /api/auth/github/callback` получает `code`/`state` в query-строке —
это, по определению OAuth Authorization Code, чувствительные
одноразовые креденшлы. Уровни защиты:

- **Код-уровневая гарантия (реализована и протестирована сейчас).**
  `web_main.py` запускает Uvicorn с `access_log=False` — access-лог с
  query-строкой никогда не пишется этим процессом. `web/app.py`'s
  `create_app()` дополнительно отключает логгер `"uvicorn.access"`
  (defense-in-depth) на случай запуска через голый `uvicorn`
  CLI/конфиг (`uvicorn web.app:create_app --factory`), где access-лог
  включён по умолчанию. Ответы `/api/auth/github/callback` (успех и
  ошибки одинаково) несут `Referrer-Policy: no-referrer` и
  `Cache-Control: no-store`.
- **Production-требование к reverse-proxy/ingress (ОБЯЗАТЕЛЬНОЕ,
  документируется здесь, физически НЕ проверяется кодом этого
  приложения).** ASGI-приложение не может задним числом запретить
  вышестоящему reverse-proxy/ingress/балансировщику логировать сырой
  входящий URI ДО того, как запрос дойдёт до этого сервера. Поэтому
  при развёртывании production reverse-proxy/ingress ОБЯЗАН либо (a)
  полностью отключить access-логирование для маршрута
  `/api/auth/github/callback`, либо (b) логировать только путь без
  query-строки / редактировать (redact) `code` и `state`. Это
  требование безопасности, а не опциональная настройка
  производительности — Stage 6B не является стадией развёртывания
  (Caddy/Traefik сознательно не настраиваются здесь), но при первом
  реальном production-развёртывании это должно быть физически
  проверено (просмотром реальной конфигурации/логов прокси), а не
  просто задокументировано. Дополнительно production ingress должен
  rate-limit'ить `/api/auth/github/login` — defense-in-depth поверх
  описанного выше database-уровневого предела, никогда не замена ему.

### Связывание Telegram- и GitHub-аккаунтов (Stage 6C)

Аутентифицированный web-пользователь (вошедший через GitHub) может явно
связать свой аккаунт с существующим Telegram-аккаунтом того же человека —
после этого канонический пользователь один: `users.id` Telegram-стороны
(**Telegram UUID переживает связывание всегда** — именно он остаётся
владельцем всех документов/настроек/RAG-данных, а не GitHub-сторона).

- **Как это работает для пользователя:**
  1. На сайте (после входа через GitHub) — `POST /api/link/telegram/start`
     (валидная сессия + CSRF) возвращает диплинк вида
     `https://t.me/<bot_username>?start=link_<секрет>` и время его
     истечения.
  2. Пользователь открывает диплинк в Telegram и нажимает Start —
     существующий allowlist-гейт (`utils.access_control`) и резолв
     Telegram UUID отрабатывают как обычно, ДО обработки `link_...`.
  3. Бот отвечает одним сообщением: успех, «уже связано» или общая
     ошибка (без деталей — исключает перебор состояния аккаунта).
  4. После успешного связывания существующие web-сессии GitHub-стороны
     **отзываются** (не переносятся) — **необходим повторный вход через
     GitHub**; новая сессия резолвится уже в переживший Telegram UUID.
- **Секрет** — `secrets.token_urlsafe(32)` (256 бит), генерируется и
  хешируется (SHA-256) только в прикладном слое (`app/telegram_link.py`);
  в PostgreSQL попадает исключительно дайджест
  (`telegram_link_attempts.link_secret_hash`) — сырой секрет НИКОГДА не
  сохраняется, не логируется и не появляется в тексте исключений. TTL —
  10 минут по умолчанию (`TELEGRAM_LINK_ATTEMPT_TTL_SECONDS`, максимум 15).
  Повторный `POST /api/link/telegram/start` атомарно заменяет предыдущий
  незавершённый запрос (не накапливает строки).
- **Итог связывания** (`app/telegram_link.py`, `db/telegram_link.py`):
  GitHub-привязка (`github_accounts`) переносится на Telegram UUID;
  все web-сессии и сама строка попытки связывания GitHub-стороны
  удаляются; пустой (identity-only) `users`-ряд GitHub-стороны удаляется.
  Владение документами/настройками/данными в Qdrant НИКОГДА не
  переписывается — они физически принадлежат Telegram UUID с самого
  начала, поэтому никакого переноса не требуется.
- **Отклонение** (общая формулировка в Telegram, без деталей — исключает
  перебор состояния): диплинк недействителен/истёк/уже использован;
  Telegram-аккаунт уже связан с ДРУГИМ GitHub-аккаунтом; GitHub-сторона
  уже связана с ДРУГИМ Telegram-аккаунтом; GitHub-привязка исчезла
  (отвязана параллельно); GitHub-сторона не «чистый» identity-аккаунт
  (несёт собственные документы/настройки — слияние было бы неоднозначным
  и данные должны остаться доступными, а не молча потеряться).
- **`POST /api/unlink/github`** (валидная сессия + CSRF) отвязывает
  GitHub от текущего канонического пользователя:
  - если у пользователя есть Telegram-привязка — GitHub-привязка
    убирается, пользователь и все его данные остаются;
  - если Telegram-привязки нет и данных (документов/настроек) тоже нет —
    аккаунт GitHub-only полностью удаляется (нечего терять);
  - если Telegram-привязки нет, но данные ЕСТЬ — запрос атомарно
    отклоняется (HTTP 409): данные никогда не остаются недостижимыми.
  - Успешная отвязка очищает cookie сессии/CSRF в ответе; при отклонении
    (409) сессия и cookie не трогаются.
- **Порядок блокировок** (защита от deadlock между конкурентными
  попытками связывания/отвязки/выпуска сессии для одного и того же
  пользователя; Stage 6C corrective pass, independent-audit MAJOR 2) —
  advisory-блокировка нужного flow (когда применима: redemption —
  `pg_advisory_xact_lock(telegram_user_id)`; unlink —
  `pg_advisory_xact_lock(-github_user_id)`; создание попытки — без
  advisory-блокировки) → `github_accounts` (при нескольких строках — с
  явным `ORDER BY`) → `users` → policy/admission-синглтон, когда требуется
  (`web_session_policy` при выпуске сессии; `github_oauth_admission` при
  успешной отвязке) → мутация строки `telegram_link_attempts` — ВСЕГДА
  ПОСЛЕДНЕЙ и всегда одним атомарным оператором (`INSERT ... ON CONFLICT`/
  `DELETE ... RETURNING`), никогда отдельным предварительным `SELECT ...
  FOR UPDATE`; ни одна операция не блокирует `github_accounts`/`users`, а
  затем ждёт уже существующей строки `telegram_link_attempts` — см.
  `db/telegram_link.py`'s docstring и `tests/test_stage6c_lock_ordering.py`
  (реальный PostgreSQL, реальные потоки и детерминированное наблюдение
  порядка SQL-операторов, доказательство отсутствия deadlock для
  критичных сценариев гонки).
- **`TELEGRAM_BOT_USERNAME`** (`.env`) — имя бота для диплинка;
  валидируется изолированным web-safe модулем (`telegram_link_config.py`,
  никогда не импортирует `TELEGRAM_BOT_TOKEN`). В отличие от остальных
  обязательных креденшлов адаптера — **отсутствие/некорректность НЕ
  останавливает запуск** web-адаптера и не ломает GitHub-логин/`/api/me`/
  logout: `POST /api/link/telegram/start` в этом случае просто отвечает
  общим HTTP 503.

## Режимы

| Команда        | Описание |
|----------------|----------|
| `/mode text`   | Текстовый диалог по Python (LLM_PROVIDER: Anthropic Claude по умолчанию, либо OpenAI) |
| `/mode voice`  | Голосовые ответы (Whisper + TTS) |
| `/mode rag`    | Ответы по базе знаний с указанием источников |
| `/mode vision` | Анализ изображений (код, ошибки, схемы) |

Режим можно переключать кнопками: команда `/mode` без аргумента открывает клавиатуру выбора.

## Команды

- `/help` — справка по боту
- `/mode` — текущий режим и кнопки переключения (или `/mode text` / `voice` / `rag` / `vision`)
- `/voice` — выбор голоса для TTS (alloy, echo, nova, fable, onyx, shimmer)
- `/reset` — очистка истории диалога
- `/stats` — статистика базы знаний (RAG)
- `/image <описание>` — генерация изображения по описанию (DALL-E 3)

Генерация изображений также срабатывает по фразам вроде «Нарисуй…», «Создай изображение…» в любом режиме.

---

## Адаптация бота под другие задачи

Бота можно использовать не только как тьютора по Python, а под любую тематику: другой язык программирования, предмет, корпоративная база знаний, поддержка и т.п. Достаточно двух вещей:

### 1. База знаний (RAG)

- Встроенная база знаний — ровно четыре версионируемых файла в `data/documents/`, перечисленные явным списком в `config.BUILTIN_REFERENCE_FILES` (см. ниже) — это НЕ произвольное сканирование папки: случайный `.txt`/`.md`-файл, оставшийся в `data/documents/`, в индекс не попадёт.
- Отправьте документ боту в Telegram (`.txt`, `.md`, `.pdf`, `.docx`), чтобы добавить СВОИ материалы — он сохранится в `data/documents/uploads/` (файл + сопроводительный `.meta.json` с оригинальным именем) и будет проиндексирован индивидуально, независимо от встроенного списка. Такой документ **приватный**: его видит и может использовать в RAG-ответах только тот пользователь (по внутреннему UUID, см. выше), который его загрузил — другим пользователям бота он не виден. Строка в PostgreSQL (`documents`) хранит владение/каталог; физический файл + `.meta.json` остаются источником содержимого, Qdrant — производный индекс.
- При старте бот автоматически сверяет Qdrant со встроенными файлами (изменившиеся или новые — без лишних повторных обращений к эмбеддингам при неизменном содержимом).
- В режиме **`/mode rag`** ответы строятся на основе: встроенной базы знаний (общая, видна всем пользователям) плюс собственных приватных документов запрашивающего пользователя — приватные документы других пользователей никогда не участвуют в поиске. В конце сообщения указывается источник (имя файла). Команда `/stats` считает документы по той же границе видимости: общая база + только ваши собственные загрузки.

Чем полнее и аккуратнее база документов, тем точнее ответы в режиме RAG.

#### Векторное хранилище (Qdrant)

С этой версии RAG использует **Qdrant** (локальный persistent-режим, `qdrant-client`) вместо ChromaDB:

- Данные хранятся локально в `data/qdrant/` — отдельный Qdrant-сервер не требуется, сетевые обращения к нему отсутствуют. **No Qdrant API key or Qdrant server is required in local mode.**
- Qdrant — это **производное (rebuildable) состояние**, а не источник истины. Источник истины — версионируемые Markdown-файлы, перечисленные в `config.BUILTIN_REFERENCE_FILES` (см. ниже) и, для загруженных через Telegram документов, пара «физический файл + `.meta.json`» в `data/documents/uploads/`.
- Полная переиндексация из исходников (без обращения к устаревшему ChromaDB) выполняется вручную, ТОЛЬКО как модуль (прямой запуск файла не работает — см. docstring скрипта):
  ```
  python -m scripts.rebuild_qdrant            # dry-run: только проверка и подсчёт, без изменений
  python -m scripts.rebuild_qdrant --apply     # реальная переиндексация (не разрушает существующий индекс: сверяет с источником, недостающее/изменившееся переиндексирует, лишнее удаляет только после успеха всех документов)
  ```
  Dry-run (без `--apply`) не требует никаких провайдерских ключей (`TELEGRAM_BOT_TOKEN`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`) и не требует файла `.env` — он только читает исходники на диске (встроенные Markdown-файлы + `.meta.json`-сайдкары загрузок) и ничего не пишет: ни в Qdrant, ни в `bot.log`. `--apply` требует обычной конфигурации эмбеддингов/провайдера (`OPENAI_API_KEY`), поскольку реально обращается к OpenAI Embeddings и мутирует Qdrant.
- Старое локальное хранилище ChromaDB (`data/chroma_db/`), если оно осталось от предыдущей версии, **не мигрируется автоматически** и ни для чего в рантайме больше не используется. После проверки работы Qdrant его можно удалить вручную.
- Владение приватными документами хранится в Qdrant как `owner_user_uuid` (канонический внутренний UUID) — не как числовой Telegram id. Коллекция названа с суффиксом `_uuid_v1` (`rag/constants.py`): это отдельная, «чистая» коллекция, не смешивающая старые записи с числовым владельцем и новые — с UUID. Старая коллекция (если существовала) физически остаётся на диске нетронутой.
- Если в `data/documents/uploads/` остались сайдкары старой (v2, числовой Telegram-владелец) схемы, переведите их на v3 ПЕРЕД обычной переиндексацией:
  ```
  python -m scripts.migrate_sidecars_v2_to_v3            # dry-run
  python -m scripts.migrate_sidecars_v2_to_v3 --apply     # реальная миграция сайдкаров (резолвит/создаёт внутренних пользователей в PostgreSQL, физический файл не трогает)
  python -m scripts.rebuild_qdrant --apply                # обычная переиндексация уже мигрированных (v3) сайдкаров
  ```

#### Встроенная база знаний по Python

`config.BUILTIN_REFERENCE_FILES` явно перечисляет четыре версионируемых (закоммиченных в репозиторий) справочных файла на английском языке в `data/documents/`: `python-fundamentals.md`, `functions-classes-errors.md`, `testing-debugging.md`, `async-python-and-apis.md`. Они демонстрируют кросс-языковой RAG (материалы на английском, диалог с ботом — на русском) и служат отправной точкой базы знаний; при адаптации под другую тематику их можно заменить своими материалами, обновив список в `config.py`.

### 2. Системные промпты

Текстовый и RAG-режимы задаются системными промптами. Их можно слегка поправить под свою задачу:

| Файл | Назначение |
|------|------------|
| **`app/tutor.py`** | Промпт для обычного текстового режима (строка с «Ты — персональный тьютор по Python…»). Замените роль и инструкции на свои (например, «тьютор по JavaScript», «ответы по внутренней документации»). |
| **`rag/query.py`** | Промпты для режима RAG: описание роли и правил ответа по контексту (`_generate_rag_response`) и fallback, когда по базе ответа нет (`_fallback_response`). Подстройте формулировки под вашу тематику и стиль ответов. |

Менять код логики не обязательно — достаточно обновить тексты промптов и наполнить `data/documents/` нужными материалами.

---

## Структура проекта

- `main.py` — точка входа Telegram-бота, подключение обработчиков, индексация RAG, инициализация/закрытие DB-движка
- `bot.py` — экземпляр бота (pyTelegramBotAPI)
- `web_main.py` — точка входа web-адаптера (Stage 6A, отдельный процесс от `main.py`)
- `web_config.py` — настройки только для web-адаптера (`SESSION_SECRET_KEY` и др.), не требуется для Telegram-бота
- `config.py` — настройки, пути, режимы (без Telegram-credential — см. `telegram_config.py`)
- `telegram_config.py` — валидация `TELEGRAM_BOT_TOKEN`, импортируется только Telegram-адаптером (`bot.py`)
- `handlers/` — start, text, voice, image, document_upload (тонкие Telegram-адаптеры; резолвят внутренний UUID сразу после проверки доступа)
- `web/` — тонкий FastAPI-адаптер: app (фабрика приложения), routes (включая `POST /api/link/telegram/start`/`POST /api/unlink/github`, Stage 6C), dependencies (централизованная проверка текущего пользователя/CSRF), cookies, csrf, schemas (Stage 6A) + github_oauth (GitHub OAuth login/callback, Stage 6B) — без бизнес-логики, весь резолвинг пользователя идёт через `app/auth_session.py`/`app/github_identity.py`/`app/telegram_link.py`
- `app/` — Telegram/Web-независимый прикладной слой: tutor (оркестрация диалога), session (состояние диалога — история/pending-image эфемерны в памяти, mode/voice — durable в PostgreSQL), documents (транзакция загрузки/индексации документа), identity (резолв Telegram id → внутренний UUID), auth_session (жизненный цикл серверной web-сессии: создание/резолв/отзыв, включая GitHub-race-safe выпуск — Stage 6A/6C), github_identity (резолв GitHub id → внутренний UUID, Stage 6B), oauth_transaction (state/PKCE-транзакция GitHub-логина, Stage 6B), telegram_link (генерация/хеширование bearer-секрета связывания, старт/redemption/отвязка — Stage 6C; сырой секрет никогда не покидает этот модуль и `web/routes.py`/`handlers/start.py`)
- `db/` — слой PostgreSQL: settings (DATABASE_URL, без credential-зависимостей), base/models (SQLAlchemy ORM), engine (ленивый sync-движок), identity (race-safe резолв/создание пользователя, чтение профиля по UUID, проверка наличия Telegram-привязки), preferences (mode/voice upsert), documents (каталог владения документами), auth_sessions (хранение web-сессий — только SHA-256 дайджест токена, включая race-safe GitHub-выпуск сессии — Stage 6A/6C), github_identity (race-safe резолв/создание пользователя по GitHub id, Stage 6B), oauth_transactions (короткоживущая одноразовая OAuth-транзакция + database-authoritative admission control/rate limit, Stage 6B), telegram_link (создание/redemption/отвязка попытки связывания под скорректированным порядком блокировок — Stage 6C)
- `github_oauth_config.py` — настройки только для GitHub-логина (`GITHUB_CLIENT_ID`/`SECRET`/`REDIRECT_URI` и др., Stage 6B), не требуется для Telegram-бота
- `telegram_link_config.py` — настройки только для связывания Telegram/GitHub (`TELEGRAM_BOT_USERNAME` и др., Stage 6C); никогда не импортирует `TELEGRAM_BOT_TOKEN`; отсутствие/некорректность значения не ломает запуск web-адаптера
- `alembic/`, `alembic.ini` — миграции схемы PostgreSQL (`alembic upgrade head`)
- `services/` — text_llm (провайдер-фасад), anthropic_client, openai_client, stt, tts, vision, image_generation, github_oauth_client (HTTP-клиент token exchange + `/user`, Stage 6B)
- `rag/` — index (Qdrant, владение по `owner_user_uuid`), query, loader (PDF, TXT, MD, DOCX), identity (стабильные ID документов/чанков + валидация канонического UUID), sidecar (метаданные загруженных документов, схема v3)
- `scripts/` — `rebuild_qdrant.py` (полная переиндексация Qdrant из исходников), `migrate_sidecars_v2_to_v3.py` (разовый перевод legacy-сайдкаров с Telegram-id на внутренний UUID)
- `utils/` — logging, helpers (Telegram-утилиты: скачивание файлов, strip_markdown, очистка файлов), access_control (fail-closed allowlist по Telegram id)
- `data/documents/` — версионируемая база знаний RAG (`.md`, закоммичены) + `uploads/` (загруженные через Telegram документы, в репозиторий не коммитятся)
- `data/qdrant/` — локальное хранилище Qdrant (в репозиторий не коммитится, полностью восстановимо через `python -m scripts.rebuild_qdrant`)
- `data/generated_images/` — сгенерированные DALL-E изображения
- `.env.example` — шаблон переменных окружения

## PostgreSQL

Постоянное хранилище для канонической идентичности пользователей, привязки
Telegram-аккаунтов, настроек (mode/voice) и каталога владения документами.
Не используется как хранилище содержимого документов (это по-прежнему
физический файл + `.meta.json`) и не хранит историю переписки (она остаётся
эфемерной, в памяти процесса).

Таблицы (`db/models.py`, миграция `alembic/versions/0001_initial_schema.py`):

| Таблица | Назначение |
|---|---|
| `users` | Канонический внутренний пользователь (`id UUID PK`) |
| `telegram_accounts` | Привязка Telegram id → `users.id` (unique в обе стороны) |
| `user_preferences` | `mode`/`voice` пользователя (durable) |
| `documents` | Каталог владения загруженными документами (`status`: pending/active) |
| `web_sessions` | Серверные web-сессии (Stage 6A, `alembic/versions/0002_web_sessions.py`): `session_token_hash` (SHA-256 браузерного bearer-токена, PK) → `users.id`, `expires_at`, `revoked_at` |
| `github_accounts` | Привязка числового GitHub id → `users.id` (Stage 6B, `alembic/versions/0003_github_oauth.py`), структурно как `telegram_accounts` |
| `github_oauth_transactions` | Короткоживущая одноразовая OAuth-транзакция GitHub-логина (Stage 6B): `state_hash` (SHA-256 дайджест `state`, PK), `code_verifier` (PKCE), `expires_at` (индексирован для cleanup). Claim — атомарный `DELETE ... RETURNING`: заявленная/просроченная строка удаляется, а не помечается — таблица никогда не содержит «мёртвых» строк, и каждая существующая строка учитывается в потолке `github_oauth_admission` |
| `github_oauth_admission` | Singleton-строка (`id=1`) глобального admission control для `/api/auth/github/login` (Stage 6B corrective pass #1): `window_start`, `starts_in_window` — фиксированное 60-секундное окно rate limit, читается/обновляется под `SELECT ... FOR UPDATE` в той же транзакции, что cleanup+вставка новой OAuth-транзакции |
| `telegram_link_attempts` | Короткоживущая одноразовая попытка связывания Telegram/GitHub (Stage 6C, `alembic/versions/0004_telegram_link_attempts.py`): `web_user_id UUID PK` (FK → `users.id`, `ON DELETE RESTRICT` — явно, не подразумеваемый) → `link_secret_hash` (SHA-256 дайджест bearer-секрета, UNIQUE, никогда не сырое значение), `expires_at` (индексирован). Не более одной строки на пользователя — повторный запрос атомарно заменяет предыдущий |

Схема создаётся ТОЛЬКО через Alembic — приложение не создаёт таблицы
самостоятельно при старте:

```
alembic upgrade head
```

`DATABASE_URL` (пример в `.env.example`) использует диалект
`postgresql+psycopg://` — один драйвер (`psycopg` v3) обслуживает и
синхронный, и асинхронный доступ. Импорт `config.py`/`db/*.py`/`app/*.py`
не требует `TELEGRAM_BOT_TOKEN` — только Telegram-адаптер (`bot.py`) и
всё, что от него зависит, требует валидный токен при старте.

## Зависимости

Основные: pyTelegramBotAPI, openai, anthropic, qdrant-client, langchain-core, langchain-openai, langchain-community, langchain-text-splitters, pypdf, docx2txt, pydub, aiofiles, aiohttp, SQLAlchemy, psycopg, alembic, fastapi, uvicorn.

Подробный список: `requirements.txt`.

## Логирование

Логи пишутся в консоль и в файл `bot.log`. Для подробного вывода в `.env` укажите `LOG_LEVEL=DEBUG`.
