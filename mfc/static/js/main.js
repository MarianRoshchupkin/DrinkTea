// Simple Bar
// new SimpleBar(document.querySelector(".gpt__dialog"), {
//     autoHide: false,
//     scrollbarMaxSize: 70
// })

function addMessageToChat(message, isUser) {
    // const simpleBar = document.querySelector('.simplebar-wrapper');
    const chatBox = document.querySelector('.gpt__dialog');
    const messageDiv = document.createElement('div');
    const messageP = document.createElement('p');

    if (isUser) {
        messageDiv.classList.add('gpt__dialog__question')
        messageP.classList.add('gpt__dialog__desc', 'gpt__dialog__question__desc')
        messageP.textContent = message;
        messageDiv.append(messageP)
    }
    else{
        messageDiv.classList.add('gpt__dialog__answer')
        messageP.classList.add('gpt__dialog__desc', 'gpt__dialog__answer__desc')
        messageP.textContent = message;
        messageDiv.append(messageP)
    }
    chatBox.append(messageDiv);

    // Прокручиваем вниз, чтобы видеть новые сообщения
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Функция для ответа на различные запросы

// Обработчик отправки формы
const messageForm = document.getElementById('messageForm')
messageForm.addEventListener('submit', function (event) {
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
