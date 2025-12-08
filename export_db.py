#!/usr/bin/env python3
"""
Скрипт для экспорта базы данных в SQL дамп или JSON файл.
Использование:
    python export_db.py --format sql --output backup.sql
    python export_db.py --format json --output backup.json
"""

import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime


def export_to_sql(db_path: Path, output_path: Path) -> None:
    """Экспортирует базу данных в SQL дамп."""
    conn = sqlite3.connect(db_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in conn.iterdump():
            f.write(f'{line}\n')
    
    conn.close()
    print(f"База данных экспортирована в SQL формат: {output_path}")


def export_to_json(db_path: Path, output_path: Path) -> None:
    """Экспортирует базу данных в JSON файл."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Получаем список всех таблиц
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    data = {}
    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        data[table] = [dict(row) for row in rows]
        print(f"Экспортировано {len(rows)} записей из таблицы {table}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    conn.close()
    print(f"База данных экспортирована в JSON формат: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Экспорт базы данных')
    parser.add_argument(
        '--db',
        type=str,
        default='yt_admin.db',
        help='Путь к базе данных (по умолчанию: yt_admin.db)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['sql', 'json'],
        default='sql',
        help='Формат экспорта: sql или json (по умолчанию: sql)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Путь к выходному файлу (по умолчанию: backup_YYYYMMDD_HHMMSS.{format})'
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Ошибка: База данных {db_path} не найдена!")
        return
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = 'sql' if args.format == 'sql' else 'json'
        output_path = Path(f'backup_{timestamp}.{extension}')
    
    if args.format == 'sql':
        export_to_sql(db_path, output_path)
    else:
        export_to_json(db_path, output_path)
    
    print(f"\nРазмер файла: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()

