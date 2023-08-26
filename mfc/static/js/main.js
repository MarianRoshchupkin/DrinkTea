function addMessageToChat(message, isUser) {
        const chatBox = document.getElementById('chatBox');
        const messageDiv = document.createElement('div');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        messageDiv.textContent = message;
        chatBox.appendChild(messageDiv);

        // Прокручиваем вниз, чтобы видеть новые сообщения
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Функция для ответа на различные запросы

    // Обработчик отправки формы
document.getElementById('messageForm').addEventListener('submit', function (event) {
        event.preventDefault(); // Предотвращаем отправку формы
        const messageInput = document.getElementById('messageInput');
        const userMessage = messageInput.value.trim().toLowerCase();
        if (userMessage == "") {
            return
        }
        addMessageToChat(userMessage, true);
        const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        let botResponse = fetch(this.action, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                "X-CSRFToken": csrftoken,
                'Accept': 'application/json'
            },
            body: JSON.stringify(userMessage)
        })
            .then(response => response.json())  // Corrected this line
            .then(botResponse => {// Here you can work with the actual response data
                addMessageToChat(botResponse.response, false);  // Assuming this function adds messages to your chat
            })
            .catch(error => {
                console.error('Error:', error);
            });
        // Очищаем поле ввода после отправки сообщения
        messageInput.value = '';
    });