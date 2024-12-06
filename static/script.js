document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = {
        amount: document.getElementById('amount').value,
        latitude: document.getElementById('latitude').value,
        longitude: document.getElementById('longitude').value,
        city_population: document.getElementById('city_population').value,
        merchant_latitude: document.getElementById('merchant_latitude').value,
        merchant_longitude: document.getElementById('merchant_longitude').value,
        category: document.getElementById('category').value
    };
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Display the result
            document.getElementById('result').innerHTML = result.message;
            document.getElementById('result').className = result.is_fraud ? 'fraud' : 'legitimate';
        } else {
            // Display error
            document.getElementById('result').innerHTML = result.message || 'Error making prediction';
            document.getElementById('result').className = 'error';
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = 'Error making prediction';
        document.getElementById('result').className = 'error';
    }
});
