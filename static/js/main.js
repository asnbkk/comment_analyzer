// Загрузка статистики
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.youtube) {
            document.getElementById('channels').textContent = formatNumber(data.youtube.channels);
            document.getElementById('videos').textContent = formatNumber(data.youtube.videos);
            document.getElementById('comments').textContent = formatNumber(data.youtube.comments);
            document.getElementById('analyzed').textContent = formatNumber(data.youtube.analyzed);
            document.getElementById('toxic').textContent = formatNumber(data.youtube.toxic);
        }
    } catch (error) {
        console.error('Ошибка загрузки статистики:', error);
    }
}

// Форматирование чисел
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    
    // Обновление статистики каждые 30 секунд
    setInterval(loadStats, 30000);
});

