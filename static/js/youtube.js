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

// Загрузка каналов
async function loadChannels() {
    try {
        const response = await fetch('/api/channels');
        const channels = await response.json();
        
        const tbody = document.getElementById('channelsTableBody');
        
        if (channels.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem; color: var(--text-secondary);">Нет каналов</td></tr>';
            return;
        }
        
        tbody.innerHTML = channels.map(channel => `
            <tr onclick="window.location.href='/youtube/channel/${channel.channel_id}'">
                <td>
                    ${channel.thumbnail_url ? `<img src="${channel.thumbnail_url}" alt="${channel.title}" class="channel-thumbnail">` : ''}
                    <span class="channel-name">${channel.title}</span>
                </td>
                <td>${formatNumber(channel.videos)}</td>
                <td>${formatNumber(channel.comments)}</td>
                <td>${formatNumber(channel.analyzed)}</td>
                <td>${formatNumber(channel.toxic)}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Ошибка загрузки каналов:', error);
        document.getElementById('channelsTableBody').innerHTML = 
            '<tr><td colspan="5" style="text-align: center; padding: 2rem; color: var(--text-secondary);">Ошибка загрузки данных</td></tr>';
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    loadChannels();
});

