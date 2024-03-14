function readURL(input) {
  if (input.files && input.files[0]) {
    let reader = new FileReader();
    reader.onload = function (e) {
      $('#input-image')
        .attr('src', e.target.result)
        .width(256)
        .height(256)
        .attr("style", "display:block");
      sessionStorage.setItem("pokemon_image", String(e.target.result));
    };
    reader.readAsDataURL(input.files[0]);
  }
}