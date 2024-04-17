document.addEventListener('DOMContentLoaded', function() {
    // Function to handle form submission for generating dataset
    //Generate_Dataset Script
    function handleDatasetGenerationForm(event) {
        event.preventDefault(); // Prevent the default form submission behavior
        // Get the form data
        const formData = new FormData(this);

        // Send a POST request to the server with the form data
        fetch('/generate_dataset', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                // If the response is successful, display a success message
                showFeedbackMessage('success', 'Dataset generation completed!');
            } else {
                // If there is an error, display an error message
                showFeedbackMessage('error', 'Error generating dataset');
            }
        })
        .catch(error => {
            // If there is a network error, display an error message
            console.error('Error:', error);
            showFeedbackMessage('error', 'Network error');
        });
    }

    // Function to handle form submission for training the classifier
    //Training_Classifier_Code
    function handleClassifierTrainingForm(event) {
        event.preventDefault(); // Prevent the default form submission behavior
        // Display a confirmation dialog before training the classifier
        if (confirm('Are you sure you want to train the classifier?')) {
            // Send a POST request to the server to trigger classifier training
            fetch('/train_classifier', {
                method: 'POST'
            })
            .then(response => {
                if (response.ok) {
                    // If the response is successful, display a success message
                    showFeedbackMessage('success', 'Classifier training completed!');
                } else {
                    // If there is an error, display an error message
                    showFeedbackMessage('error', 'Error training classifier');
                }
            })
            .catch(error => {
                // If there is a network error, display an error message
                console.error('Error:', error);
                showFeedbackMessage('error', 'Network error');
            });
        }
    }

    // Function to handle form submission for uploading image
    //Uploading_Image_Project
    function handleImageUploadForm(event) {
        event.preventDefault(); // Prevent the default form submission behavior
        // Get the form data
        const formData = new FormData(this);

        // Send a POST request to the server with the form data
        fetch('/upload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                // If the response is successful, display a success message
                showFeedbackMessage('success', 'Image uploaded successfully!');
            } else {
                // If there is an error, display an error message
                showFeedbackMessage('error', 'Error uploading image');
            }
        })
        .catch(error => {
            // If there is a network error, display an error message
            console.error('Error:', error);
            showFeedbackMessage('error', 'Network error');
        });
    }

    // Function to display feedback messages
    function showFeedbackMessage(type, message) {
        const feedbackDiv = document.getElementById('feedback-message');
        feedbackDiv.textContent = message;
        feedbackDiv.className = type;
        feedbackDiv .style.display = 'block';
        setTimeout(function() {
            feedbackDiv.style.display = 'none';
        }, 3000);
    }

    // Get the generate dataset form element
    const generateDatasetForm = document.getElementById('generate-dataset-form');
    // Add event listener for form submission
    generateDatasetForm.addEventListener('submit', handleDatasetGenerationForm);

    // Get the train classifier form element
    const trainClassifierForm = document.getElementById('train-classifier-form');
    // Add event listener for form submission
    trainClassifierForm.addEventListener('submit', handleClassifierTrainingForm);

    // Get the upload image form element
    const uploadImageForm = document.getElementById('upload-image-form');
    // Add event listener for form submission
    uploadImageForm.addEventListener('submit', handleImageUploadForm);
});
