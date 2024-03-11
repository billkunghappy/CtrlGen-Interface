// For the sliders
var ctrl_len_slider;
var ctrl_token_slider;

function SetThumbText(slider, slider_id_str, single_range = false){
    var value = slider.value();
    if (!single_range){
        // For single range slider, we don't do this
        $(slider_id_str + ' > div:eq(0)').text(value[0]);
    }
    $(slider_id_str + ' > div:eq(1)').text(value[1]);
}

function InitSlider(){
    console.log("init the sliders");
    // Initialize The Sliders
    ctrl_len_slider = rangeSlider(document.querySelector('#ctrl-length-slider'), {
        "min": 1,
        "max": 64,
        "step" : 1,
        "value" : [10, 20], // Default value
        "onInput": (value, userInteraction) => {
            console.log(value);
            // Set the first thumbs max value, and the second thums min value
            $('#ctrl-length-slider > input:eq(0)').attr({'max': value[1]});
            $('#ctrl-length-slider > input:eq(1)').attr({'min': value[0]});
            // Update the text value
            SetThumbText(ctrl_len_slider, '#ctrl-length-slider');
        }
    });
    SetThumbText(ctrl_len_slider, '#ctrl-length-slider');
    // At start, disable it
    ctrl_len_slider.disabled(true);

    ctrl_token_slider = rangeSlider(document.querySelector('#ctrl-token-slider'), {
        "min": 1,
        "max": 64,
        "step" : 2,
        "value" : [0, 1], // For one value slider, first value needs to be 0
        thumbsDisabled: [true, false],
        rangeSlideDisabled: true,
        "onInput": (value, userInteraction) => {
            // Update the text value
            SetThumbText(ctrl_token_slider, '#ctrl-token-slider', single_range = true);
        }
    });
    SetThumbText(ctrl_token_slider, '#ctrl-token-slider', single_range = true);
    // At start, disable it
    ctrl_token_slider.disabled(true);
}

async function reset_slider(value = null, min = null, max = null, disable = false){
    if (min != null){
        ctrl_len_slider.min(min);
        console.log("Set slider min " + String(min));
    }
    if (max != null){
        ctrl_len_slider.max(max);
        console.log("Set slider max " + String(max));
    }
    if (value != null){
        ctrl_len_slider.value(value);
        console.log("Set slider value " + String(value));
    }
    ctrl_len_slider.disabled(disable);
    return true;
}