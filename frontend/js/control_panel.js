function resetHMMCtrl(
    keyword = "",
    banword = "",
    length_unit = "none",
    length = [10, 25],
    instruct = ""
  ){
    console.log("Reset Control");
    $("#ctrl-keyword").val(keyword);
    $("#ctrl-banword").val(banword);
    $("#ctrl-length_unit").val(length_unit);
    // Set length range at slider
    reset_slider(length, min = null, max = null, disable = true);
    $("#ctrl-instruct").val(instruct);

    // Reverse animate
    $('#ctrl-keyword').parents(".card-body").removeClass('control');
    $('#ctrl-banword').parents(".card-body").removeClass('control');
    $('#ctrl-length_unit').parents(".card-body").removeClass('control');
    $('#ctrl-instruct').parents(".card-body").removeClass('control');
}

// Check if apply control
function apply_control(){
    if ($("#ctrl-switch").is(':checked')){
        return true;
    }
    return false;
}

// Event Listeners
$( "#control-panel-btn" ).on( "click", function() {
    if ($("#ctrl-switch").is(':checked')){
        // Already checked, unchecked it
        // $("#ctrl-switch").prop('checked', false)
    }
    else{
        // $("#ctrl-switch").prop('checked', true)
    }
    $("#ctrl-switch").click();
});

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
$('#ctrl-banword').on('input', function() {
    if ($('#ctrl-banword').val().trim() != ""){
        $('#ctrl-banword').parents(".card-body").addClass('control');
    }
    else{
        $('#ctrl-banword').parents(".card-body").removeClass('control');
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

// Token Control
var token_slider_output_dict = {}; // When the slider has value changed, we update it 
var token_insert_curIndex = -1;
var token_insert_text_len = -1;

function init_token_output_list(){
    for (i = ctrl_token_range_min; i < ctrl_token_range_max; i+=ctrl_token_range_step) {
        token_slider_output_dict[i] = "Y".repeat(i);
    }
}
function update_token_output_list(suggestions){
    var cnt_idx = 0;
    for (i = ctrl_token_range_min; i < ctrl_token_range_max; i+=ctrl_token_range_step) {
        token_slider_output_dict[i] = suggestions[cnt_idx]['original'];
        cnt_idx += 1;
    }
}

function show_token_control(){
    if (apply_token_control){
        $("#ctrl-token-collapse").removeClass("slideright");
    }
}

function disable_token_control(){
    ctrl_token_slider.disabled(true);
    $("#ctrl-token-collapse").addClass("slideright");
    change_token_control_status("loading");
}

function token_loaded_success(){
    // TODO:
    console.log("Successfully load the token results!");
    change_token_control_status("success");
    ctrl_token_slider.disabled(false);
    // Reset value to start
    ctrl_token_slider.value([0,ctrl_token_range_min]);
    // Dont need to remove slideright class here since before we load the models output, we already show it
}

function token_loaded_failed(){
    // TODO:
    alert("Fail to load the token results!");
    change_token_control_status("fail");
    disable_token_control();
}

// Animation
function change_token_control_status(status){
    if (status == "success"){
        $("#ctrl-token-success").show();
        $("#ctrl-token-fail").hide();
        $("#ctrl-token-spinner").hide();
    }
    else if (status == "fail"){
        $("#ctrl-token-success").hide();
        $("#ctrl-token-fail").show();
        $("#ctrl-token-spinner").hide();
    }
    else if (status == "loading"){
        $("#ctrl-token-success").hide();
        $("#ctrl-token-fail").hide();
        $("#ctrl-token-spinner").show();
    }
}
change_token_control_status("loading");