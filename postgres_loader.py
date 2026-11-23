"""Read one repeatable, read-only PostgreSQL snapshot into the CSV pipeline."""

import csv
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Reserve room for downstream copies, feature matrices and trained models in the 32 GiB runner.
_FRAME_BUDGET = 4 * 1024**3
_FIXED_WIDTH = {"smallint", "integer", "bigint", "numeric", "real", "double precision", "boolean"}
PG_CONNECTION_ENV = ("PGHOST", "PGPORT", "PGUSER", "PGDATABASE")
PG_ENV = (*PG_CONNECTION_ENV, "PGPASSWORD")


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple
    target: tuple
    target_columns: tuple


@dataclass
class Table:
    schema: str
    name: str
    columns: tuple
    column_types: dict = field(default_factory=dict)
    comments: dict = field(default_factory=dict)
    primary_key: tuple = ()
    foreign_keys: tuple = ()


def _identifier(value):
    # Both parts of a qualified name come from the catalog and are quoted separately.
    return '"' + value.replace('"', '""') + '"'


def _relation(table):
    return _identifier(table.schema) + "." + _identifier(table.name)


def _remaining(deadline):
    left = 120 if deadline is None else deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("Истёк бюджет загрузки PostgreSQL.")
    return left


def _set_timeout(conn, deadline):
    milliseconds = max(1, int(1000 * min(120, _remaining(deadline))))
    conn.execute("SELECT pg_catalog.set_config('statement_timeout', %s, true)", (str(milliseconds),))


def _catalog(conn):
    column_rows = conn.execute("""
        SELECT c.table_schema, c.table_name, c.column_name, c.data_type,
               pg_catalog.col_description(r.oid, c.ordinal_position)
        FROM information_schema.columns AS c
        JOIN pg_catalog.pg_namespace AS n ON n.nspname = c.table_schema
        JOIN pg_catalog.pg_class AS r ON r.relnamespace = n.oid AND r.relname = c.table_name
        WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
          AND c.table_schema !~ '^pg_(temp|toast)' AND NOT r.relispartition
          AND r.relkind IN ('r', 'p', 'v')
        ORDER BY c.table_schema, c.table_name, c.ordinal_position
    """)
    tables = {}
    for schema, name, column, dtype, comment in column_rows:
        key = schema, name
        if key not in tables:
            tables[key] = Table(schema, name, ())
        table = tables[key]
        table.columns += (column,)
        table.column_types[column] = dtype
        if comment:
            table.comments[column] = comment
    keys = conn.execute("""
        SELECT n.nspname, r.relname, c.contype, rn.nspname, rr.relname,
               ARRAY(SELECT a.attname FROM unnest(c.conkey) WITH ORDINALITY k(num,pos)
                     JOIN pg_catalog.pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.num
                     ORDER BY k.pos),
               ARRAY(SELECT a.attname FROM unnest(c.confkey) WITH ORDINALITY k(num,pos)
                     JOIN pg_catalog.pg_attribute a ON a.attrelid=c.confrelid AND a.attnum=k.num
                     ORDER BY k.pos)
        FROM pg_catalog.pg_constraint c
        JOIN pg_catalog.pg_class r ON r.oid=c.conrelid
        JOIN pg_catalog.pg_namespace n ON n.oid=r.relnamespace
        LEFT JOIN pg_catalog.pg_class rr ON rr.oid=c.confrelid
        LEFT JOIN pg_catalog.pg_namespace rn ON rn.oid=rr.relnamespace
        WHERE c.contype IN ('p','f') AND n.nspname NOT IN ('pg_catalog','information_schema')
    """)
    for schema, name, kind, ref_schema, ref_name, cols, ref_cols in keys:
        table = tables.get((schema, name))
        if table is None:
            continue
        if kind == "p":
            table.primary_key = tuple(cols)
        else:
            table.foreign_keys += (ForeignKey(tuple(cols), (ref_schema, ref_name), tuple(ref_cols)),)
    return tables


def _word(text, name):
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I) is not None


def _local_sample(task_path, task_text):
    folder = task_path.parent.resolve()
    adjacent = folder / "sample_submission.csv"
    if adjacent.resolve().is_relative_to(folder) and adjacent.is_file():
        return adjacent
    candidates = set()
    for match in re.finditer(r"[\w./\\-]*sample_submission\.csv\b", task_text, re.I):
        path = (folder / match.group()).resolve()
        if path.is_relative_to(folder) and path.is_file():
            candidates.add(path)
    if len(candidates) > 1:
        raise ValueError("Условие указывает несколько локальных sample_submission.csv.")
    if candidates:
        return candidates.pop()

    # Task bundles can put their input in data/ or data/<case>/. Ignore hidden
    # workspaces and use the database template if multiple bundles share a root.
    for pattern in ("*/sample_submission.csv", "*/*/sample_submission.csv"):
        for path in folder.glob(pattern):
            parts = path.relative_to(folder).parts[:-1]
            if any(p.startswith(".") or p in {"venv", "node_modules", "__pycache__"} for p in parts):
                continue
            resolved = path.resolve()
            if resolved.is_relative_to(folder) and resolved.is_file():
                candidates.add(resolved)
                if len(candidates) > 1:
                    return None
    return next(iter(candidates), None)


def _sample_table(tables, text):
    names = {"sample_submission", "submission_sample", "submission_template", "sample"}
    scores = []
    for table in tables.values():
        normalized = table.name.casefold()
        declared = re.search(
            r"(?:sample(?:_submission)?|submission template|шаблон(?: ответа)?)"
            r"\s*(?:table|таблица)?\s*[:=]\s*[`\"]?(?:\w+\.)?" +
            re.escape(table.name) + r"(?!\w)", text, re.I)
        if declared or normalized in names or re.search(r"(?:sample.*submission|submission.*sample)", normalized):
            scores.append(((bool(declared), int(_word(text, table.schema + "." + table.name)),
                            int(_word(text, table.name)), int(normalized == "sample_submission")), table))
    if not scores:
        raise ValueError("Не найден локальный шаблон или таблица sample_submission.")
    score = max(item[0] for item in scores)
    selected = [t for s, t in scores if s == score]
    if len(selected) != 1:
        raise ValueError("Неоднозначная таблица шаблона PostgreSQL.")
    return selected[0]


_ORDINALS = {"row_number", "row_num", "row_index", "row_ordinal", "row_order", "row_position",
             "sample_order", "sample_position", "submission_order", "ordinal", "ordinal_position"}


def _sample_order(table, text):
    context = r"sample|submission|шаблон|сабмит|порядок строк"
    lines = text.splitlines()
    order_lines = [line for i, line in enumerate(lines) if re.search(context, line, re.I) or
                   (i and re.search(context, lines[i - 1], re.I) and
                    re.match(r"\s*[`]*order\s+by\b", line, re.I))]
    explicit = set()
    for col in table.columns:
        name = r'[`"]?' + re.escape(col) + r'[`"]?(?!\w)'
        pattern = (r"(?:order(?:ed)?\s+by|sorted\s+by|сортиров\w*\s+по|отсортир\w*\s+по)\s+" +
                   name + r"(?:\s+(?:in\s+)?(desc(?:ending)?|asc(?:ending)?|по убыванию|по возрастанию))?")
        for line in order_lines:
            for match in re.finditer(pattern, line, re.I):
                if re.search(r"\b(?:not|without|не|без)\s*$", line[:match.start()], re.I):
                    continue
                if re.match(r"\s*(?:[-+*/%(:]|collate\b)", line[match.end():], re.I):
                    raise ValueError("Выражение ORDER BY шаблона не является простой колонкой.")
                direction = (match.group(1) or "").lower()
                explicit.add((col, direction.startswith("desc") or direction == "по убыванию"))
            direct = re.search(r"(?:по |в порядке )(возрастани[юя]|убывани[юя])\s+" + name, line, re.I)
            if direct:
                explicit.add((col, direct.group(1).lower().startswith("убыв")))
    ordinal = [col for col in table.columns if
               table.column_types.get(col) in ("integer", "smallint", "bigint", "numeric") and
               (col.lower() in _ORDINALS or re.search(
                   r"(?:original|sample|submission|csv) row (?:order|number|position)|"
                   r"поряд(?:ок|ковый номер) строк[и]? (?:шаблона|образца|csv)",
                   table.comments.get(col, ""), re.I))]
    if len(explicit) == 1:
        key, descending = explicit.pop()
        return key, descending, ordinal
    if not explicit and len(ordinal) == 1:
        return ordinal[0], False, ordinal
    raise ValueError("Порядок sample_submission не определён однозначно. Нужен исходный CSV, "
                     "колонка порядка строк или заданный в task.md ORDER BY.")


def _role(table, role, text):
    names = {"train": ("train", "training", "labeled", "labelled"),
             "test": ("test", "inference", "scoring", "holdout", "unlabeled", "unlabelled")}[role]
    normalized = table.name.casefold()
    match = bool(set(re.split(r"\W+|_", normalized)) & set(names))
    label = r"(?:train(?:ing)?|обучающ\w*)" if role == "train" else r"(?:test|inference|тестов\w*)"
    description = re.search(r"\b" + label + r"\s+(?:data\s+)?(?:table|таблица)\s*[:=]?\s*[`\"]?(?:\w+\.)?" +
                            re.escape(table.name) + r"(?!\w)", text, re.I)
    return 4 * bool(description) + 2 * (normalized in names) + int(match)


def _declared(columns, text, label):
    return [c for c in columns if re.search(
        r"(?<!\w)(?:" + label + r")\s*[:=—-]?\s*[`\"]?" +
        re.escape(c) + r"(?!\w)", text, re.I)]


def _binding(train, test, sample_columns, text, ordinals=()):
    train_cols, test_cols = set(train.columns), set(test.columns)
    declared = _declared(train.columns, text,
                         r"target(?: column)?|label column|таргет|целевая (?:колонка|переменная)")
    if len(declared) > 1:
        return None
    ids = [c for c in sample_columns if c in test_cols and c not in declared]
    # An explicitly ordered identifier may itself be named row_number or ordinal.
    ordinary_ids = [c for c in ids if c not in ordinals]
    declared_ids = _declared(ids, text, r"id column|identifier|идентификатор|ключ строки")
    if len(declared_ids) == 1:
        ids = declared_ids
    elif len(ids) > 1 and len(test.primary_key) == 1 and test.primary_key[0] in ids:
        ids = list(test.primary_key)
    elif ordinary_ids:
        ids = ordinary_ids
    if len(ids) > 1:
        ids = [c for c in ids if c.lower() == "id" or c.lower().endswith(("_id", "_key"))]
    if len(ids) != 1:
        return None
    identifier = ids[0]
    differences = train_cols - test_cols
    answers = set(sample_columns) - {identifier, *ordinals}
    if declared:
        target = declared[0]
    elif len(differences) == 1:
        target = next(iter(differences))
    elif len(answers) == 1 and answers <= differences:
        target = next(iter(answers))
    elif differences:
        mentioned = [c for c in differences if _word(text, c)]
        if len(mentioned) != 1:
            return None
        target = mentioned[0]
    elif len(answers) == 1 and answers <= train_cols:
        target = next(iter(answers))
    else:
        return None
    shared = (train_cols & test_cols) - {identifier, target}
    all_features = (train_cols | test_cols) - {identifier, target}
    overlap = len(shared) / len(all_features) if all_features else 0.0
    return identifier, target, overlap


def _choose_pair(tables, sample, sample_columns, text, ordinals=()):
    choices = []
    available = [t for t in tables.values() if t is not sample and t.name.lower() not in
                 {"sample_submission", "submission_sample", "submission_template", "data_dictionary"}]
    roles = {(t.schema, t.name): (_role(t, "train", text), _role(t, "test", text)) for t in available}
    for train in available:
        for test in available:
            if train is test:
                continue
            found = _binding(train, test, sample_columns, text, ordinals)
            if found is None:
                continue
            id_col, target, overlap = found
            train_role = roles[(train.schema, train.name)][0]
            test_role = roles[(test.schema, test.name)][1]
            if (target in test.columns or not overlap) and not (train_role and test_role):
                continue
            score = (train_role + test_role, int(_word(text, train.schema + "." + train.name)) +
                     int(_word(text, test.schema + "." + test.name)), target not in test.columns,
                     train.schema == test.schema, overlap)
            choices.append((score, (train, test, id_col, target)))
    if not choices:
        raise ValueError("Не удалось определить совместимые train/test, target и ID по схеме и шаблону.")
    highest = max(s for s, _ in choices)
    best = [value for s, value in choices if s == highest]
    if len(best) != 1:
        raise ValueError("Несколько равноправных пар train/test. Укажите таблицы в task.md.")
    return best[0]


def _fetch_size(rows, bytes_read):
    """Choose a bounded batch using observed row width on this export."""
    return max(1, min(32768, (4 * 1024**2 * rows) // max(bytes_read, 1)))


def _export(conn, table, columns, path, deadline, *, order=None, descending=False, check_order=False,
            memory_budget=_FRAME_BUDGET, id_column=None, progress=None):
    fields = list(columns)
    if order and order not in fields:
        fields.append(order)
    sql = "SELECT " + ", ".join(map(_identifier, fields)) + " FROM " + _relation(table)
    if order:
        sql += " ORDER BY " + _identifier(order) + (" DESC" if descending else " ASC")
    previous = object()
    index = fields.index(order) if check_order else None
    fixed_bytes = sum(8 if c != id_column and table.column_types.get(c) in _FIXED_WIDTH else 72
                      for c in columns)
    count = written = 0
    started = last_update = time.monotonic()

    def report(complete):
        if progress:
            progress(path.stem, {"rows": count, "bytes": written, "complete": complete,
                                "seconds": round(time.monotonic() - started, 3)})

    report(False)
    _set_timeout(conn, deadline)
    with conn.cursor(name="tyrell_export") as cursor, path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        cursor.execute(sql)
        batch = 1024
        last_timeout = started
        while True:
            left = _remaining(deadline)
            now = time.monotonic()
            if left < 120 and now - last_timeout >= min(10.0, left / 4):
                _set_timeout(conn, deadline)
                last_timeout = now
            rows = cursor.fetchmany(batch)
            if not rows:
                break
            if check_order:
                for row in rows:
                    key = row[index]
                    if key is None or key == previous:
                        raise ValueError("Колонка порядка строк шаблона содержит NULL или повторы.")
                    previous = key
                writer.writerows(row[:len(columns)] for row in rows)
            else:
                writer.writerows(rows)
            count += len(rows)
            written = stream.tell()
            size = written + count * fixed_bytes
            # Adapt to measured row width. Fewer network round trips for narrow tables;
            # wide text/JSON stays in small batches instead of buffering an entire result.
            batch = _fetch_size(count, size)
            if time.monotonic() - last_update >= 10:
                report(False)
                last_update = time.monotonic()
            if size > memory_budget:
                raise ValueError("Данные превышают бюджет загрузки в pandas для 32 ГиБ RAM "
                                 f"(таблица {_relation(table)}, прочитано {count} строк).")
        written = stream.tell()
        report(True)
        return written + count * fixed_bytes


def pg_error_reason(error):
    # Do not forward raw libpq diagnostics: they can contain connection details.
    state = getattr(error, "sqlstate", None)
    message = str(error).casefold()
    if "no password supplied" in message:
        reason = "password required by server"
    elif state == "28P01" or "password authentication failed" in message:
        reason = "password authentication failed"
    elif state == "57014":
        reason = "statement was cancelled or exceeded its timeout"
    elif state == "42501" or "permission denied" in message:
        reason = "insufficient read privileges"
    elif "connection refused" in message:
        reason = "connection refused by endpoint"
    elif "timed out" in message or "timeout expired" in message:
        reason = "connection timed out"
    elif "server closed the connection" in message:
        reason = "connection closed by server or tunnel"
    else:
        reason = type(error).__name__
    return f"{reason} (SQLSTATE {state or 'unavailable'})"


def prepare_postgres(task_path, output_dir, *, deadline=None, log=print, on_core_ready=None, on_stats=None):
    """Export once per run into a new scratch folder. Preserve source files and credentials."""
    import psycopg

    if not all(os.environ.get(name) for name in PG_CONNECTION_ENV):
        raise ValueError("Для PostgreSQL нужны PGHOST, PGPORT, PGUSER и PGDATABASE.")
    from data_tools import enrich_postgres

    task_path, output_dir = Path(task_path), Path(output_dir)
    text = task_path.read_text(encoding="utf-8-sig")
    local = _local_sample(task_path, text)
    if any((output_dir / f).exists() for f in
           ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")):
        raise ValueError("Для PostgreSQL нужен новый временный каталог.")
    started = time.monotonic()
    exports = {}
    joins = []
    join_audit = []

    def audit_join(detail):
        join_audit.append(detail)
        on_stats(join_audit=join_audit)

    def export_progress(name, detail):
        exports[name] = detail
        action = "готов" if detail["complete"] else "чтение"
        log(f"PostgreSQL {name}: {action}, {detail['rows']} строк, "
            f"{detail['bytes'] / 2**20:.1f} МиБ, {detail['seconds']:.1f} с")
        if on_stats:
            fields = {"phase": "export_" + name, "exports": exports}
            if detail["complete"] and name in ("train", "test"):
                fields[name + "_rows"] = detail["rows"]
            on_stats(**fields)

    def join_log(message):
        log(message)
        if on_stats:
            joins.append(message)
            on_stats(joins=joins)

    try:
        # libpq reads its connection credentials from the operator's PG* environment.
        with psycopg.connect(connect_timeout=max(1, min(10, math.ceil(_remaining(deadline))))) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            _set_timeout(conn, deadline)
            tables = _catalog(conn)
            sample = None
            ordinals = ()
            if local:
                with local.open(encoding="utf-8-sig", newline="") as file:
                    columns = next(csv.reader(file), [])
                log(f"Шаблон и порядок ID из CSV {local.name}.")
            else:
                sample = _sample_table(tables, text)
                order, desc, ordinals = _sample_order(sample, text)
                columns = list(sample.columns)
                log(f"Шаблон из {_relation(sample)}, порядок по {_identifier(order)} {'DESC' if desc else 'ASC'}.")
            if len(columns) < 2 or any(not c for c in columns) or len(set(columns)) != len(columns):
                raise ValueError("Некорректные имена колонок шаблона submission.")
            train, test, id_col, target = _choose_pair(tables, sample, columns, text, ordinals)
            columns = [c for c in columns if c == id_col or c not in ordinals]
            if len(columns) < 2:
                raise ValueError("Отсутствует столбец ответа в шаблоне.")
            if target in test.columns:
                _set_timeout(conn, deadline)
                sql = f"SELECT 1 FROM {_relation(test)} WHERE {_identifier(target)} IS NOT NULL LIMIT 1"
                if conn.execute(sql).fetchone() is not None:
                    raise ValueError("В тестовой таблице обнаружены известные значения целевой переменной.")
            log(f"PostgreSQL train={_relation(train)}, test={_relation(test)}, ID={id_col}, target={target}.")
            available = _FRAME_BUDGET
            if local:
                shutil.copyfile(local, output_dir / "sample_submission.csv")
                available -= local.stat().st_size
            else:
                available -= _export(conn, sample, columns, output_dir / "sample_submission.csv", deadline,
                                     order=order, descending=desc, check_order=True, id_column=id_col,
                                     progress=export_progress)
            train_cols = [c for c in train.columns if c in test.columns or c == target]
            test_cols = [c for c in test.columns if c != target and (c in train_cols or c == id_col)]
            pair_order = id_col if os.environ.get("TYRELL_PAIR_ARM") in ("automl", "qwen") else None
            available -= _export(conn, test, test_cols, output_dir / "test.csv", deadline,
                                 order=pair_order, memory_budget=available, id_column=id_col, progress=export_progress)
            _export(conn, train, train_cols, output_dir / "train.csv", deadline,
                    order=pair_order if id_col in train_cols else None,
                    memory_budget=available, id_column=id_col, progress=export_progress)
            with (output_dir / "data_dictionary.csv").open("x", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("feature", "role", "description"))
                for name in dict.fromkeys([*train_cols, id_col]):
                    role = "id" if name == id_col else "target" if name == target else "feature"
                    writer.writerow((name, role, train.comments.get(name) or test.comments.get(name, "")))
            snapshot_seconds = time.monotonic() - started
            if on_stats:
                on_stats(phase="fallback", snapshot_seconds=round(snapshot_seconds, 3),
                         data_load_seconds=round(snapshot_seconds, 3))
            if on_core_ready is not None:
                on_core_ready()
            join_start = time.monotonic()
            if on_stats:
                on_stats(phase="joins", joins=joins)
            try:
                if deadline is None or deadline > join_start:
                    enrich_postgres(conn, tables, train, test, id_col, target, text, output_dir, deadline, join_log,
                                    on_audit=audit_join if on_stats else None)
                else:
                    join_log("Достигнут лимит времени JOIN")
            finally:
                if on_stats:
                    seconds = time.monotonic() - join_start
                    on_stats(join_seconds=round(seconds, 3), data_load_seconds=round(snapshot_seconds + seconds, 3))
    except psycopg.Error as exc:
        # Connection diagnostics can contain addresses and role names.
        raise RuntimeError(f"Ошибка чтения PostgreSQL: {pg_error_reason(exc)}.") from None
