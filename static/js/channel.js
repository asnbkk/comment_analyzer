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

// Загрузка информации о канале
async function loadChannelInfo() {
    try {
        const response = await fetch(`/api/channel/${channelId}`);
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('channelInfo').innerHTML = 
                '<p style="color: var(--text-secondary);">Канал не найден</p>';
            return;
        }
        
        // Заполняем информацию о канале
        document.getElementById('channelTitle').textContent = data.title;
        
        const thumbnail = document.getElementById('channelThumbnail');
        if (data.thumbnail_url) {
            thumbnail.style.backgroundImage = `url(${data.thumbnail_url})`;
            thumbnail.style.backgroundSize = 'cover';
            thumbnail.style.backgroundPosition = 'center';
        }
        
        if (data.subscriber_count) {
            document.getElementById('channelSubscribers').textContent = 
                `${formatNumber(data.subscriber_count)} подписчиков`;
        }
        
        // Заполняем статистику
        document.getElementById('videos').textContent = formatNumber(data.stats.videos);
        document.getElementById('comments').textContent = formatNumber(data.stats.comments);
        document.getElementById('analyzed').textContent = formatNumber(data.stats.analyzed);
        document.getElementById('toxic').textContent = formatNumber(data.stats.toxic);
    } catch (error) {
        console.error('Ошибка загрузки информации о канале:', error);
    }
}

// Загрузка видео с пагинацией
let currentPage = 1;
const perPage = 20;

async function loadVideos(page = 1) {
    try {
        const response = await fetch(`/api/channel/${channelId}/videos?page=${page}&per_page=${perPage}`);
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('videosList').innerHTML = 
                '<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">Ошибка загрузки видео</div>';
            return;
        }
        
        const videosList = document.getElementById('videosList');
        
        if (data.videos.length === 0) {
            videosList.innerHTML = 
                '<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">Нет видео</div>';
            return;
        }
        
        videosList.innerHTML = data.videos.map(video => `
            <div class="video-item" onclick="window.location.href='/youtube/video/${video.video_id}'">
                <div class="video-thumbnail">
                    ${video.thumbnail_url ? `<img src="${video.thumbnail_url}" alt="${video.title}">` : ''}
                </div>
                <div class="video-info">
                    <h4 class="video-title">${video.title}</h4>
                    <div class="video-meta">
                        <span>${formatDate(video.published_at)}</span>
                        <span>${formatNumber(video.view_count)} просмотров</span>
                    </div>
                    <div class="video-stats">
                        <span>Комментариев: ${formatNumber(video.comments_fetched || 0)}</span>
                        <span>Проанализировано: ${formatNumber(video.analyzed || 0)}</span>
                        <span>Токсичных: ${formatNumber(video.toxic || 0)}</span>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Обновляем пагинацию
        updatePagination(data.pagination);
        currentPage = page;
    } catch (error) {
        console.error('Ошибка загрузки видео:', error);
        document.getElementById('videosList').innerHTML = 
            '<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">Ошибка загрузки данных</div>';
    }
}

function updatePagination(pagination) {
    const paginationEl = document.getElementById('pagination');
    
    if (pagination.pages <= 1) {
        paginationEl.innerHTML = '';
        return;
    }
    
    let html = '<div class="pagination-controls">';
    
    // Кнопка "Назад"
    if (pagination.page > 1) {
        html += `<button onclick="loadVideos(${pagination.page - 1})" class="pagination-btn">← Назад</button>`;
    }
    
    // Номера страниц
    const startPage = Math.max(1, pagination.page - 2);
    const endPage = Math.min(pagination.pages, pagination.page + 2);
    
    if (startPage > 1) {
        html += `<button onclick="loadVideos(1)" class="pagination-btn">1</button>`;
        if (startPage > 2) {
            html += `<span class="pagination-dots">...</span>`;
        }
    }
    
    for (let i = startPage; i <= endPage; i++) {
        if (i === pagination.page) {
            html += `<button class="pagination-btn active">${i}</button>`;
        } else {
            html += `<button onclick="loadVideos(${i})" class="pagination-btn">${i}</button>`;
        }
    }
    
    if (endPage < pagination.pages) {
        if (endPage < pagination.pages - 1) {
            html += `<span class="pagination-dots">...</span>`;
        }
        html += `<button onclick="loadVideos(${pagination.pages})" class="pagination-btn">${pagination.pages}</button>`;
    }
    
    // Кнопка "Вперед"
    if (pagination.page < pagination.pages) {
        html += `<button onclick="loadVideos(${pagination.page + 1})" class="pagination-btn">Вперед →</button>`;
    }
    
    html += '</div>';
    paginationEl.innerHTML = html;
}

function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('ru-RU', { year: 'numeric', month: 'long', day: 'numeric' });
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    loadChannelInfo();
    loadVideos(1);
});
