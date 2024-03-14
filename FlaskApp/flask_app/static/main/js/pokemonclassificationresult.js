const inputImage = document.getElementById('input-image');
const tryAgainButton = document.getElementById('try-again');
$(document).ready(() => {
    inputImage.src = sessionStorage.getItem("pokemon_image");
});

tryAgainButton.addEventListener('click', () => {
    window.location.href = '/';
});
