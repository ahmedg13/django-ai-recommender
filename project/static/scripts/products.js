document.addEventListener('DOMContentLoaded', function() {
    const minPriceSlider = document.getElementById('min-price');
    const maxPriceSlider = document.getElementById('max-price');
    const minPriceDisplay = document.getElementById('min-price-display');
    const maxPriceDisplay = document.getElementById('max-price-display');

    // Price range slider interaction
    function updatePriceRange() {
        if (parseInt(minPriceSlider.value) > parseInt(maxPriceSlider.value)) {
            [minPriceSlider.value, maxPriceSlider.value] = [maxPriceSlider.value, minPriceSlider.value];
        }
        
        minPriceDisplay.textContent = minPriceSlider.value;
        maxPriceDisplay.textContent = maxPriceSlider.value;
    }

    minPriceSlider.addEventListener('input', updatePriceRange);
    maxPriceSlider.addEventListener('input', updatePriceRange);

    // Wishlist toggle
    const wishlistIcons = document.querySelectorAll('.product-wishlist-icon');
    wishlistIcons.forEach(icon => {
        icon.addEventListener('click', function() {
            this.classList.toggle('active');
            const heartIcon = this.querySelector('i');
            heartIcon.classList.toggle('fas');
            heartIcon.classList.toggle('far');
        });
    });
});