document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('prediction-results');

    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();

        
            if (typeof jsonDataToSend === 'undefined') {
                console.error('Error: La variable jsonDataToSend no está definida.');
                resultsContainer.textContent = 'Error: No se encontraron los datos para enviar.';
                return;
            }

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(jsonDataToSend)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Resultados del backend:', data);
                // Actualiza el contenido del contenedor de resultados con la respuesta del backend
                if (resultsContainer) {
                    // Formatea los resultados como desees (ejemplo básico en JSON stringificado)
                    resultsContainer.textContent = JSON.stringify(data, null, 2);
                    // O puedes acceder a propiedades específicas de 'data' y mostrarlas
                    // Ejemplo: resultsContainer.textContent = `Predicción: ${data.prediction}`;
                }
            })
            .catch(error => {
                console.error('Error al enviar los datos o recibir la respuesta:', error);
                if (resultsContainer) {
                    resultsContainer.textContent = 'Error al obtener la predicción.';
                }
            });
        });
    } else {
        console.error('Error: No se encontró el formulario con el ID "prediction-form".');
    }
});