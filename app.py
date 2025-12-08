#!/usr/bin/env python3
"""
Веб-приложение для анализа комментариев из социальных сетей
"""

from flask import Flask, render_template, jsonify
import sqlite3
from pathlib import Path

app = Flask(__name__)
DB_PATH = Path("yt_admin.db")


def get_db_connection():
    """Получить соединение с базой данных"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    """Главная страница"""
    return render_template("index.html")


@app.route("/youtube")
def youtube():
    """Страница YouTube"""
    return render_template("youtube.html")


@app.route("/youtube/channel/<channel_id>")
def channel_detail(channel_id):
    """Страница канала"""
    return render_template("channel.html", channel_id=channel_id)


@app.route("/youtube/video/<video_id>")
def video_detail(video_id):
    """Страница видео"""
    return render_template("video.html", video_id=video_id)


@app.route("/api/stats")
def get_stats():
    """Получить статистику по YouTube"""
    conn = get_db_connection()
    
    try:
        # Статистика по видео
        video_count = conn.execute("SELECT COUNT(*) as count FROM video").fetchone()["count"]
        
        # Статистика по комментариям
        comment_count = conn.execute("SELECT COUNT(*) as count FROM comment_details").fetchone()["count"]
        
        # Статистика по анализу токсичности
        analyzed_count = conn.execute("SELECT COUNT(*) as count FROM comment_analysis").fetchone()["count"]
        
        # Статистика по токсичным комментариям
        toxic_count = conn.execute(
            "SELECT COUNT(*) as count FROM comment_analysis WHERE toxicity_score >= 0.7"
        ).fetchone()["count"]
        
        # Статистика по каналам
        channel_count = conn.execute("SELECT COUNT(*) as count FROM channel").fetchone()["count"]
        
        return jsonify({
            "youtube": {
                "channels": channel_count,
                "videos": video_count,
                "comments": comment_count,
                "analyzed": analyzed_count,
                "toxic": toxic_count
            }
        })
    finally:
        conn.close()


@app.route("/api/channels")
def get_channels():
    """Получить список каналов со статистикой"""
    conn = get_db_connection()
    
    try:
        channels = conn.execute("""
            SELECT 
                c.channel_id,
                c.title,
                c.thumbnail_url,
                c.subscriber_count,
                COUNT(DISTINCT v.video_id) as videos,
                COUNT(DISTINCT cd.comment_id) as comments,
                COUNT(DISTINCT ca.comment_id) as analyzed,
                COUNT(DISTINCT CASE WHEN ca.toxicity_score >= 0.7 THEN ca.comment_id END) as toxic
            FROM channel c
            LEFT JOIN video v ON c.channel_id = v.channel_id
            LEFT JOIN comment_details cd ON v.video_id = cd.video_id
            LEFT JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
            GROUP BY c.channel_id, c.title, c.thumbnail_url, c.subscriber_count
            ORDER BY c.title
        """).fetchall()
        
        result = []
        for ch in channels:
            result.append({
                "channel_id": ch["channel_id"],
                "title": ch["title"],
                "thumbnail_url": ch["thumbnail_url"],
                "subscriber_count": ch["subscriber_count"],
                "videos": ch["videos"],
                "comments": ch["comments"],
                "analyzed": ch["analyzed"],
                "toxic": ch["toxic"]
            })
        
        return jsonify(result)
    finally:
        conn.close()


@app.route("/api/channel/<channel_id>")
def get_channel_detail(channel_id):
    """Получить детальную информацию о канале"""
    conn = get_db_connection()
    
    try:
        channel = conn.execute(
            "SELECT * FROM channel WHERE channel_id = ?", (channel_id,)
        ).fetchone()
        
        if not channel:
            return jsonify({"error": "Channel not found"}), 404
        
        # Статистика по видео
        videos = conn.execute(
            "SELECT COUNT(*) as count FROM video WHERE channel_id = ?", (channel_id,)
        ).fetchone()["count"]
        
        # Статистика по комментариям
        comments = conn.execute("""
            SELECT COUNT(*) as count 
            FROM comment_details cd
            JOIN video v ON cd.video_id = v.video_id
            WHERE v.channel_id = ?
        """, (channel_id,)).fetchone()["count"]
        
        # Статистика по анализу
        analyzed = conn.execute("""
            SELECT COUNT(*) as count 
            FROM comment_analysis ca
            JOIN comment_details cd ON ca.comment_id = cd.comment_id
            JOIN video v ON cd.video_id = v.video_id
            WHERE v.channel_id = ?
        """, (channel_id,)).fetchone()["count"]
        
        # Токсичные комментарии
        toxic = conn.execute("""
            SELECT COUNT(*) as count 
            FROM comment_analysis ca
            JOIN comment_details cd ON ca.comment_id = cd.comment_id
            JOIN video v ON cd.video_id = v.video_id
            WHERE v.channel_id = ? AND ca.toxicity_score >= 0.7
        """, (channel_id,)).fetchone()["count"]
        
        return jsonify({
            "channel_id": channel["channel_id"],
            "title": channel["title"],
            "description": channel["description"],
            "thumbnail_url": channel["thumbnail_url"],
            "subscriber_count": channel["subscriber_count"],
            "video_count": channel["video_count"],
            "view_count": channel["view_count"],
            "stats": {
                "videos": videos,
                "comments": comments,
                "analyzed": analyzed,
                "toxic": toxic
            }
        })
    finally:
        conn.close()


@app.route("/api/channel/<channel_id>/videos")
def get_channel_videos(channel_id):
    """Получить список видео канала с пагинацией"""
    from flask import request
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    conn = get_db_connection()
    
    try:
        # Проверяем существование канала
        channel = conn.execute(
            "SELECT channel_id FROM channel WHERE channel_id = ?", (channel_id,)
        ).fetchone()
        
        if not channel:
            return jsonify({"error": "Channel not found"}), 404
        
        # Общее количество видео
        total = conn.execute(
            "SELECT COUNT(*) as count FROM video WHERE channel_id = ?", (channel_id,)
        ).fetchone()["count"]
        
        # Получаем видео с пагинацией (оптимизированный запрос с CTE)
        offset = (page - 1) * per_page
        videos = conn.execute("""
            WITH video_stats AS (
                SELECT 
                    cd.video_id,
                    COUNT(DISTINCT cd.comment_id) as comments_fetched,
                    COUNT(DISTINCT ca.comment_id) as analyzed,
                    COUNT(DISTINCT CASE WHEN ca.toxicity_score >= 0.7 THEN ca.comment_id END) as toxic
                FROM comment_details cd
                LEFT JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
                GROUP BY cd.video_id
            )
            SELECT 
                v.video_id,
                v.title,
                v.thumbnail_url,
                v.published_at,
                v.view_count,
                v.comments_count,
                COALESCE(vs.comments_fetched, 0) as comments_fetched,
                COALESCE(vs.analyzed, 0) as analyzed,
                COALESCE(vs.toxic, 0) as toxic
            FROM video v
            LEFT JOIN video_stats vs ON v.video_id = vs.video_id
            WHERE v.channel_id = ?
            ORDER BY v.published_at DESC
            LIMIT ? OFFSET ?
        """, (channel_id, per_page, offset)).fetchall()
        
        result = []
        for vid in videos:
            result.append({
                "video_id": vid["video_id"],
                "title": vid["title"],
                "thumbnail_url": vid["thumbnail_url"],
                "published_at": vid["published_at"],
                "view_count": vid["view_count"],
                "comments_count": vid["comments_count"],
                "comments_fetched": vid["comments_fetched"],
                "analyzed": vid["analyzed"],
                "toxic": vid["toxic"]
            })
        
        return jsonify({
            "videos": result,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        })
    finally:
        conn.close()


@app.route("/api/video/<video_id>")
def get_video_detail(video_id):
    """Получить детальную информацию о видео"""
    conn = get_db_connection()
    
    try:
        video = conn.execute(
            "SELECT * FROM video WHERE video_id = ?", (video_id,)
        ).fetchone()
        
        if not video:
            return jsonify({"error": "Video not found"}), 404
        
        # Получаем информацию о канале
        channel = conn.execute(
            "SELECT channel_id, title, thumbnail_url FROM channel WHERE channel_id = ?", 
            (video["channel_id"],)
        ).fetchone()
        
        # Статистика по комментариям
        comments_fetched = conn.execute(
            "SELECT COUNT(*) as count FROM comment_details WHERE video_id = ?", (video_id,)
        ).fetchone()["count"]
        
        analyzed = conn.execute("""
            SELECT COUNT(*) as count 
            FROM comment_analysis ca
            JOIN comment_details cd ON ca.comment_id = cd.comment_id
            WHERE cd.video_id = ?
        """, (video_id,)).fetchone()["count"]
        
        toxic = conn.execute("""
            SELECT COUNT(*) as count 
            FROM comment_analysis ca
            JOIN comment_details cd ON ca.comment_id = cd.comment_id
            WHERE cd.video_id = ? AND ca.toxicity_score >= 0.7
        """, (video_id,)).fetchone()["count"]
        
        return jsonify({
            "video_id": video["video_id"],
            "title": video["title"],
            "description": video["description"],
            "thumbnail_url": video["thumbnail_url"],
            "video_url": video["video_url"],
            "published_at": video["published_at"],
            "view_count": video["view_count"],
            "like_count": video["like_count"],
            "comments_count": video["comments_count"],
            "channel": {
                "channel_id": channel["channel_id"],
                "title": channel["title"],
                "thumbnail_url": channel["thumbnail_url"]
            },
            "stats": {
                "comments_fetched": comments_fetched,
                "analyzed": analyzed,
                "toxic": toxic
            }
        })
    finally:
        conn.close()


@app.route("/api/video/<video_id>/comments")
def get_video_comments(video_id):
    """Получить комментарии видео, отсортированные по токсичности, с ответами"""
    from flask import request
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    conn = get_db_connection()
    
    try:
        # Проверяем существование видео
        video = conn.execute(
            "SELECT video_id FROM video WHERE video_id = ?", (video_id,)
        ).fetchone()
        
        if not video:
            return jsonify({"error": "Video not found"}), 404
        
        # Общее количество родительских комментариев с анализом
        total = conn.execute("""
            SELECT COUNT(*) as count 
            FROM comment_details cd
            JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
            WHERE cd.video_id = ? AND cd.parent_id IS NULL
        """, (video_id,)).fetchone()["count"]
        
        # Получаем родительские комментарии с пагинацией, отсортированные по токсичности
        offset = (page - 1) * per_page
        parent_comments = conn.execute("""
            SELECT 
                cd.comment_id,
                cd.text_display,
                cd.text_original,
                cd.author_display_name,
                cd.published_at,
                cd.like_count,
                ca.toxicity_score
            FROM comment_details cd
            JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
            WHERE cd.video_id = ? AND cd.parent_id IS NULL
            ORDER BY ca.toxicity_score DESC, cd.published_at DESC
            LIMIT ? OFFSET ?
        """, (video_id, per_page, offset)).fetchall()
        
        result = []
        for parent in parent_comments:
            # Получаем все ответы к этому комментарию
            replies = conn.execute("""
                SELECT 
                    cd.comment_id,
                    cd.text_display,
                    cd.text_original,
                    cd.author_display_name,
                    cd.published_at,
                    cd.like_count,
                    ca.toxicity_score
                FROM comment_details cd
                LEFT JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
                WHERE cd.parent_id = ?
                ORDER BY cd.published_at ASC
            """, (parent["comment_id"],)).fetchall()
            
            # Формируем ответы
            replies_list = []
            for reply in replies:
                replies_list.append({
                    "comment_id": reply["comment_id"],
                    "text": reply["text_display"] or reply["text_original"] or "",
                    "author": reply["author_display_name"] or "Аноним",
                    "published_at": reply["published_at"],
                    "like_count": reply["like_count"],
                    "toxicity_score": reply["toxicity_score"] if reply["toxicity_score"] is not None else 0.0
                })
            
            result.append({
                "comment_id": parent["comment_id"],
                "text": parent["text_display"] or parent["text_original"] or "",
                "author": parent["author_display_name"] or "Аноним",
                "published_at": parent["published_at"],
                "like_count": parent["like_count"],
                "toxicity_score": parent["toxicity_score"],
                "replies": replies_list
            })
        
        return jsonify({
            "comments": result,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        })
    finally:
        conn.close()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

