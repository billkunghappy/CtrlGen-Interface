function resetHMMCtrl(
    keyword = "",
    length_unit = "none",
    length = [10, 25],
    instruct = ""
  ){
    console.log("Reset Control");
    $("#ctrl-keyword").val(keyword);
    $("#ctrl-length_unit").val(length_unit);
    // Set length range at slider
    reset_slider(length, min = null, max = null, disable = true);
    $("#ctrl-instruct").val(instruct);

    // Reverse animate
    $('#ctrl-keyword').parents(".card-body").removeClass('control');
    $('#ctrl-length_unit').parents(".card-body").removeClass('control');
    $('#ctrl-instruct').parents(".card-body").removeClass('control');
}

// Event Listeners
$( "#ctrl-reset-btn" ).on( "click", function() {
    resetHMMCtrl();
  } );
$('#ctrl-keyword').on('input', function() {
    if ($('#ctrl-keyword').val().trim() != ""){
        $('#ctrl-keyword').parents(".card-body").addClass('control');
    }
    else{
        $('#ctrl-keyword').parents(".card-body").removeClass('control');
    }
});
$('#ctrl-instruct').on('input', function() {
    if ($('#ctrl-instruct').val().trim() != ""){
        $('#ctrl-instruct').parents(".card-body").addClass('control');
    }
    else{
        $('#ctrl-instruct').parents(".card-body").removeClass('control');
    }
});
