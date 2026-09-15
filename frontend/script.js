document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const probabilityValue = document.getElementById('probability-value');
    const decisionBadge = document.getElementById('decision-badge');
    const progressCircle = document.querySelector('.score-circle .progress');

    // Circle properties
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    progressCircle.style.strokeDashoffset = circumference;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Reset UI state
        errorContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
        submitBtn.classList.add('loading');
        
        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        // Convert numbers
        data.age = parseInt(data.age, 10);
        data.balance = parseFloat(data.balance);
        data.campaign = parseInt(data.campaign, 10);
        data.duration = parseFloat(data.duration);

        // Configuration: Update this URL to your Render deployment URL when deployed
        // e.g., const API_BASE_URL = 'https://your-app-name.onrender.com';
        const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:'
            ? 'http://localhost:5000'
            : 'https://optimizing-customer-response-prediction.onrender.com'; // <-- CHANGE THIS TO YOUR RENDER URL

        try {
            // Send request to Flask API
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || result.message || 'An error occurred during prediction.');
            }

            // Update UI with success result
            showResult(result.probability, result.decision);

        } catch (error) {
            console.error('Error:', error);
            errorMessage.textContent = error.message || 'Failed to connect to the prediction server. Ensure it is running.';
            errorContainer.classList.remove('hidden');
        } finally {
            submitBtn.classList.remove('loading');
        }
    });

    function showResult(probability, decision) {
        resultContainer.classList.remove('hidden');
        
        // Convert probability to percentage (0-100)
        const percentage = Math.round(probability * 100);
        
        // Animate counter
        animateValue(probabilityValue, 0, percentage, 1000);
        
        // Animate circular progress bar
        const offset = circumference - (probability * circumference);
        
        // Small delay to allow CSS transition to kick in after display:block
        setTimeout(() => {
            progressCircle.style.strokeDashoffset = offset;
            
            // Change color based on score
            if (probability >= 0.5) {
                progressCircle.style.stroke = 'var(--success)';
            } else {
                progressCircle.style.stroke = 'var(--danger)';
            }
        }, 50);

        // Update Decision Badge
        decisionBadge.textContent = decision;
        if (decision.toLowerCase().includes('do not')) {
            decisionBadge.className = 'decision-badge not-target';
        } else {
            decisionBadge.className = 'decision-badge target';
        }
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start)) + start + '%';
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
