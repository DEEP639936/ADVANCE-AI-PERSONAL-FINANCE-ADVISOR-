/**
 * AI Personal Finance Advisor v2.0 - Main JavaScript
 * =================================================
 * Features:
 * - Dark/Light theme toggle with localStorage
 * - Authentication state management
 * - Toast notifications
 * - API helpers with auth headers
 * - Loading animations
 * - INR currency support
 */

// ============================================
// THEME MANAGEMENT
// ============================================
const ThemeManager = {
    init() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme);

        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggle());
        }
    },

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);

        const icon = document.querySelector('#themeToggle i');
        if (icon) {
            icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
        // Refresh charts when theme changes
        if (typeof window.refreshChartsForTheme === 'function') {
            window.refreshChartsForTheme(theme);
        }
    },

    toggle() {
        const current = document.documentElement.getAttribute('data-theme') || 'light';
        const next = current === 'dark' ? 'light' : 'dark';
        this.setTheme(next);
    },

    getCurrent() {
        return document.documentElement.getAttribute('data-theme') || 'light';
    }
};

// ============================================
// AUTHENTICATION MANAGEMENT
// ============================================
const AuthManager = {
    token: null,
    user: null,

    init() {
        this.token = localStorage.getItem('auth_token');
        this.user = JSON.parse(localStorage.getItem('auth_user') || 'null');
        this.updateUI();
    },

    isLoggedIn() {
        return !!this.token;
    },

    setAuth(token, user) {
        this.token = token;
        this.user = user;
        localStorage.setItem('auth_token', token);
        localStorage.setItem('auth_user', JSON.stringify(user));
        this.updateUI();
    },

    clearAuth() {
        this.token = null;
        this.user = null;
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
        this.updateUI();
    },

    updateUI() {
        const authLinks = document.getElementById('authLinks');
        const userMenu = document.getElementById('userMenu');
        const protectedLinks = document.querySelectorAll('.protected-link');

        if (this.isLoggedIn() && this.user) {
            if (authLinks) authLinks.style.display = 'none';
            if (userMenu) {
                userMenu.style.display = 'flex';
                const usernameEl = userMenu.querySelector('.username');
                const avatarEl = userMenu.querySelector('.user-avatar');
                if (usernameEl) usernameEl.textContent = this.user.username;
                if (avatarEl) avatarEl.textContent = this.user.username.charAt(0).toUpperCase();
            }
            protectedLinks.forEach(el => el.style.display = 'block');
        } else {
            if (authLinks) authLinks.style.display = 'flex';
            if (userMenu) userMenu.style.display = 'none';
            protectedLinks.forEach(el => el.style.display = 'none');
        }
    },

    getAuthHeaders() {
        return this.token ? { 'Authorization': `Bearer ${this.token}` } : {};
    },

    async logout() {
        try {
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: this.getAuthHeaders()
            });
        } catch (e) {
            console.error('Logout error:', e);
        }
        this.clearAuth();
        window.location.href = '/';
    }
};

// ============================================
// TOAST NOTIFICATIONS
// ============================================
const ToastManager = {
    container: null,

    init() {
        if (!document.getElementById('toastContainer')) {
            const container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        this.container = document.getElementById('toastContainer');
    },

    show(message, type = 'info', duration = 4000) {
        if (!this.container) this.init();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type} d-flex align-items-center`;

        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };

        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.info} me-2"></i>
            <div class="flex-grow-1">${message}</div>
            <button type="button" class="btn-close-toast ms-2" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;

        this.container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(30px)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
};

// ============================================
// API HELPERS
// ============================================
async function apiGet(url) {
    const headers = {
        'Content-Type': 'application/json',
        ...AuthManager.getAuthHeaders()
    };

    const res = await fetch(url, { headers });

    if (res.status === 401) {
        AuthManager.clearAuth();
        window.location.href = '/login';
        return null;
    }

    return res.json();
}

async function apiPost(url, data) {
    const headers = {
        'Content-Type': 'application/json',
        ...AuthManager.getAuthHeaders()
    };

    const res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(data)
    });

    if (res.status === 401) {
        AuthManager.clearAuth();
        window.location.href = '/login';
        return null;
    }

    return res.json();
}

// ============================================
// LOADING OVERLAY
// ============================================
function showLoading(message = 'Loading...') {
    let overlay = document.getElementById('loadingOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.style.cssText = `
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.6); backdrop-filter: blur(5px);
            display: flex; align-items: center; justify-content: center;
            z-index: 9999; flex-direction: column; gap: 1rem;
        `;
        document.body.appendChild(overlay);
    }
    overlay.innerHTML = `
        <div class="spinner-premium"></div>
        <p style="color: #fff; font-weight: 500;">${message}</p>
    `;
    overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'none';
}

// ============================================
// CURRENCY FORMATTING - INR (Indian Rupee)
// ============================================
function formatCurrency(amount) {
    if (amount === undefined || amount === null) return '₹0.00';
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 2
    }).format(amount);
}

function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

// ============================================
// AUTH GUARD
// ============================================
function requireAuth() {
    if (!AuthManager.isLoggedIn()) {
        window.location.href = '/login';
        return false;
    }
    return true;
}

// ============================================
// INIT ON DOM READY
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    AuthManager.init();
    ToastManager.init();

    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
});


AuthManager.token = localStorage.getItem('auth_token');
AuthManager.user  = JSON.parse(localStorage.getItem('auth_user') || 'null');