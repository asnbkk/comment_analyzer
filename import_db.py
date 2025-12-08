#!/usr/bin/env python3
"""
Скрипт для импорта базы данных из SQL дампа или JSON файла.
Использование:
    python import_db.py --format sql --input backup.sql --output yt_admin.db
    python import_db.py --format json --input backup.json --output yt_admin.db
"""

import sqlite3
import json
import argparse
from pathlib import Path


def import_from_sql(input_path: Path, output_path: Path) -> None:
    """Импортирует базу данных из SQL дампа."""
    conn = sqlite3.connect(output_path)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        sql_script = f.read()
    
    conn.executescript(sql_script)
    conn.commit()
    conn.close()
    
    print(f"База данных импортирована из SQL файла: {input_path}")
    print(f"Создана база данных: {output_path}")


def import_from_json(input_path: Path, output_path: Path) -> None:
    """Импортирует базу данных из JSON файла."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    for table_name, rows in data.items():
        if not rows:
            continue
        
        # Получаем структуру таблицы
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_sql = cursor.fetchone()
        
        if not table_sql:
            print(f"Предупреждение: Таблица {table_name} не найдена в схеме. Пропускаем.")
            continue
        
        # Очищаем таблицу перед импортом
        cursor.execute(f"DELETE FROM {table_name}")
        
        # Получаем колонки таблицы
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Вставляем данные
        placeholders = ','.join(['?' for _ in columns])
        insert_sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
        
        for row in rows:
            values = [row.get(col) for col in columns]
            cursor.execute(insert_sql, values)
        
        print(f"Импортировано {len(rows)} записей в таблицу {table_name}")
    
    conn.commit()
    conn.close()
    
    print(f"База данных импортирована из JSON файла: {input_path}")
    print(f"Создана база данных: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Импорт базы данных')
    parser.add_argument(
        '--format',
        type=str,
        choices=['sql', 'json'],
        required=True,
        help='Формат импорта: sql или json'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Путь к файлу для импорта'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='yt_admin.db',
        help='Путь к выходной базе данных (по умолчанию: yt_admin.db)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: Файл {input_path} не найден!")
        return
    
    output_path = Path(args.output)
    
    if output_path.exists():
        response = input(f"База данных {output_path} уже существует. Перезаписать? (y/n): ")
        if response.lower() != 'y':
            print("Импорт отменен.")
            return
        output_path.unlink()
    
    if args.format == 'sql':
        import_from_sql(input_path, output_path)
    else:
        import_from_json(input_path, output_path)
    
    print(f"\nРазмер базы данных: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()

