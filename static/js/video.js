// Форматирование чисел (если функция не определена в main.js)
if (typeof formatNumber === 'undefined') {
    function formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
}

// Загрузка информации о видео
async function loadVideoInfo() {
    try {
        const response = await fetch(`/api/video/${videoId}`);
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('videoInfo').innerHTML = 
                '<p style="color: var(--text-secondary);">Видео не найдено</p>';
            return;
        }
        
        // Устанавливаем ссылку "Назад"
        const backLink = document.getElementById('backLink');
        if (data.channel) {
            backLink.href = `/youtube/channel/${data.channel.channel_id}`;
            backLink.textContent = `← Назад к каналу: ${data.channel.title}`;
        }
        
        // Форматируем дату
        const publishedDate = formatDate(data.published_at);
        
        // Заполняем информацию о видео
        document.getElementById('videoInfo').innerHTML = `
            <div style="display: flex; gap: 2rem; margin-bottom: 2rem;">
                <div style="flex-shrink: 0;">
                    ${data.thumbnail_url ? `<img src="${data.thumbnail_url}" alt="${data.title}" style="width: 400px; max-width: 100%; border-radius: 0.5rem;">` : ''}
                </div>
                <div style="flex: 1;">
                    <h2 class="dashboard-title" style="margin-bottom: 1rem;">${data.title}</h2>
                    <div style="margin-bottom: 1rem;">
                        <a href="${data.video_url}" target="_blank" style="color: var(--primary-color); text-decoration: none;">
                            Открыть на YouTube →
                        </a>
                    </div>
                    <div style="color: var(--text-secondary); margin-bottom: 1rem;">
                        <div>Опубликовано: ${publishedDate}</div>
                        <div>Просмотров: ${formatNumber(data.view_count)}</div>
                        ${data.like_count ? `<div>Лайков: ${formatNumber(data.like_count)}</div>` : ''}
                    </div>
                    ${data.description ? `<div style="color: var(--text-secondary); margin-top: 1rem; line-height: 1.6;">${data.description}</div>` : ''}
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">💬</div>
                    <div class="stat-content">
                        <div class="stat-value">${formatNumber(data.stats.comments_fetched)}</div>
                        <div class="stat-label">Комментариев</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🔍</div>
                    <div class="stat-content">
                        <div class="stat-value">${formatNumber(data.stats.analyzed)}</div>
                        <div class="stat-label">Проанализировано</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">⚠️</div>
                    <div class="stat-content">
                        <div class="stat-value">${formatNumber(data.stats.toxic)}</div>
                        <div class="stat-label">Токсичных</div>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Ошибка загрузки информации о видео:', error);
        document.getElementById('videoInfo').innerHTML = 
            '<p style="color: var(--text-secondary);">Ошибка загрузки данных</p>';
    }
}

function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('ru-RU', { year: 'numeric', month: 'long', day: 'numeric' });
}

// Загрузка комментариев с пагинацией
let currentCommentsPage = 1;
const commentsPerPage = 50;

async function loadComments(page = 1) {
    try {
        const response = await fetch(`/api/video/${videoId}/comments?page=${page}&per_page=${commentsPerPage}`);
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('commentsList').innerHTML = 
                '<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">Ошибка загрузки комментариев</div>';
            return;
        }
        
        const commentsList = document.getElementById('commentsList');
        
        if (data.comments.length === 0) {
            commentsList.innerHTML = 
                '<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">Нет комментариев</div>';
            return;
        }
        
        commentsList.innerHTML = data.comments.map(comment => {
            const toxicityColor = comment.toxicity_score >= 0.7 ? '#dc2626' : 
                                 comment.toxicity_score >= 0.5 ? '#f59e0b' : '#10b981';
            
            // Формируем ответы
            const repliesHtml = comment.replies && comment.replies.length > 0 ? `
                <div class="comment-replies">
                    ${comment.replies.map(reply => {
                        const replyToxicityColor = reply.toxicity_score >= 0.7 ? '#dc2626' : 
                                                  reply.toxicity_score >= 0.5 ? '#f59e0b' : '#10b981';
                        return `
                            <div class="comment-item comment-reply">
                                <div class="comment-content">
                                    <div class="comment-header">
                                        <span class="comment-author">${escapeHtml(reply.author)}</span>
                                    </div>
                                    <div class="comment-text">${escapeHtml(reply.text)}</div>
                                    <div class="comment-meta">
                                        <span>${formatDate(reply.published_at)}</span>
                                        ${reply.like_count ? `<span>👍 ${formatNumber(reply.like_count)}</span>` : ''}
                                    </div>
                                </div>
                                ${reply.toxicity_score > 0 ? `
                                    <div class="comment-toxicity" style="background-color: ${replyToxicityColor}20; color: ${replyToxicityColor};">
                                        ${(reply.toxicity_score * 100).toFixed(1)}%
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    }).join('')}
                </div>
            ` : '';
            
            return `
                <div class="comment-item comment-parent">
                    <div style="flex: 1;">
                        <div class="comment-content">
                            <div class="comment-header">
                                <span class="comment-author">${escapeHtml(comment.author)}</span>
                            </div>
                            <div class="comment-text">${escapeHtml(comment.text)}</div>
                            <div class="comment-meta">
                                <span>${formatDate(comment.published_at)}</span>
                                ${comment.like_count ? `<span>👍 ${formatNumber(comment.like_count)}</span>` : ''}
                                ${comment.replies && comment.replies.length > 0 ? `<span>💬 ${comment.replies.length} ответ${comment.replies.length === 1 ? '' : comment.replies.length < 5 ? 'а' : 'ов'}</span>` : ''}
                            </div>
                        </div>
                        ${repliesHtml}
                    </div>
                    <div class="comment-toxicity" style="background-color: ${toxicityColor}20; color: ${toxicityColor};">
                        ${(comment.toxicity_score * 100).toFixed(1)}%
                    </div>
                </div>
            `;
        }).join('');
        
        // Обновляем пагинацию комментариев
        updateCommentsPagination(data.pagination);
        currentCommentsPage = page;
    } catch (error) {
        console.error('Ошибка загрузки комментариев:', error);
        document.getElementById('commentsList').innerHTML = 
            '<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">Ошибка загрузки данных</div>';
    }
}

function updateCommentsPagination(pagination) {
    const paginationEl = document.getElementById('commentsPagination');
    
    if (pagination.pages <= 1) {
        paginationEl.innerHTML = '';
        return;
    }
    
    let html = '<div class="pagination-controls">';
    
    if (pagination.page > 1) {
        html += `<button onclick="loadComments(${pagination.page - 1})" class="pagination-btn">← Назад</button>`;
    }
    
    const startPage = Math.max(1, pagination.page - 2);
    const endPage = Math.min(pagination.pages, pagination.page + 2);
    
    if (startPage > 1) {
        html += `<button onclick="loadComments(1)" class="pagination-btn">1</button>`;
        if (startPage > 2) {
            html += `<span class="pagination-dots">...</span>`;
        }
    }
    
    for (let i = startPage; i <= endPage; i++) {
        if (i === pagination.page) {
            html += `<button class="pagination-btn active">${i}</button>`;
        } else {
            html += `<button onclick="loadComments(${i})" class="pagination-btn">${i}</button>`;
        }
    }
    
    if (endPage < pagination.pages) {
        if (endPage < pagination.pages - 1) {
            html += `<span class="pagination-dots">...</span>`;
        }
        html += `<button onclick="loadComments(${pagination.pages})" class="pagination-btn">${pagination.pages}</button>`;
    }
    
    if (pagination.page < pagination.pages) {
        html += `<button onclick="loadComments(${pagination.page + 1})" class="pagination-btn">Вперед →</button>`;
    }
    
    html += '</div>';
    paginationEl.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    loadVideoInfo();
    loadComments(1);
});

