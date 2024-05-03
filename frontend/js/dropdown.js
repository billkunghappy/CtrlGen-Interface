var dropdown_container = $('#frontend-overlay');
var numberVisibleDropdown = 3;
var dropdown_anime_speed = 150;

function vertCycleDropdownDown() {
  inAnimation = true;
  var firstItem = dropdown_container.find('.dropdown-item:first').clone(true, true);
  dropdown_container.append(firstItem);
  firstItem = null;
  dropdown_container.find('.dropdown-item.sudo-hover').removeClass("sudo-hover");
  dropdown_container.find('.dropdown-item:first').animate({ marginTop: "-50px" }, dropdown_anime_speed, function(){
    $(this).remove();
    dropdown_container.find('.dropdown-item:nth-child(2)').addClass("sudo-hover");
    suggestion_switch(dropdown_container.find('.dropdown-item:nth-child(2)'));
  });
  inAnimation = false
  return true
}

function vertCycleDropdownUp() {
  inAnimation = true;
  var lastItem = dropdown_container.find('.dropdown-item').last().clone(true, true);
  lastItem.css('margin-top', "-50px");
  dropdown_container.prepend(lastItem);
  lastItem = null;
  dropdown_container.find('.dropdown-item.sudo-hover').removeClass("sudo-hover");
  dropdown_container.find('.dropdown-item:first').animate({ marginTop: "0px" }, dropdown_anime_speed, function(){
    dropdown_container.find('.dropdown-item').last().remove();
    dropdown_container.find('.dropdown-item:nth-child(2)').addClass("sudo-hover");
    suggestion_switch(dropdown_container.find('.dropdown-item:nth-child(2)'));
  });
  inAnimation = false
  return true
  // dropdown_container.find('.dropdown-item:nth-child(2)').addClass("sudo-hover");
  // lastItem = dropdown_container.find('.dropdown-item').last().remove();
  // dropdown_container.find('.dropdown-item:nth-child(' + String(numberVisibleDropdown) +')').animate({ marginTop: "-50px" }, 300, function(){  $(this).remove(); dropdown_container.find('.dropdown-item:nth-child(2)').addClass("sudo-hover"); });
}

function emptyDropdownMenu() {
  $('#frontend-overlay').empty();
}


function openDropdownMenu(source, is_reopen=false) {
  if ($('#frontend-overlay').hasClass('hidden')){
    $('#frontend-overlay').removeClass('hidden');
    // Disable the editor until the suggestion dropdown is closed
    // Before that, remember the cursor position
    // cursor_range = quill.getSelection();
    // currentIndex = cursor_range[0] + cursor_range[1]
    quill.enable(false); 
    // quill.blur();
  }

  if (is_reopen == true) {
    logEvent(EventName.SUGGESTION_REOPEN, source);
  } else {
    logEvent(EventName.SUGGESTION_OPEN, source);
  }
}

function close_dropdown(click_item = false, abort = false){
  // console.log(click_item);
  if (hideDropdownMenu(EventSource.USER, abort = abort)){
    if (!click_item) {
      remove_to_rewrite();
    }
    else{
      // Click the item
      dropdown_container.find('.dropdown-item:nth-child(2)').click();
    }
    // Adter the dropdown is closed, enable the editor and set the cursor
    quill.enable();
    quill.focus();
    abortRewrite(abort);
    // quill.setSelection(currentIndex);
  }
  
}

function checkDropdownShown(){
  return $('#frontend-overlay').length && !$('#frontend-overlay').hasClass('hidden');
}

function hideDropdownMenu(source, abort = false) {
  if (checkDropdownShown()){
    $('#frontend-overlay').addClass('hidden');
    $('.sudo-hover').removeClass('sudo-hover');  // NOTE Do not delete; error
    if (abort){
      // Remove the suggestions
      logEvent(EventName.SUGGESTION_ABORT, EventSource.USER);
    }
    else{
      logEvent(EventName.SUGGESTION_CLOSE, source);
    }
    // console.log("Hide");
    return true;
  }
  return false;
}

function selectDropdownItem(suggestion){
  // Close dropdown menu after selecting new suggestion
  // logEvent(EventName.SUGGESTION_SELECT, EventSource.USER);
  // hideDropdownMenu(EventSource.API);
  // appendText(suggestion);
  remove_all_format();
  reset_to_rewrite();
  // Also call reset 
  try {
    resetHMMCtrl();
  } catch ({ name, message }) {
    console.log(name);
    console.log(message);
  }
  // Do not empty for metaphor generation
  if (domain != 'metaphor'){
    emptyDropdownMenu();
  }

}

function suggestion_switch(selection){
  appendText(selection.data("original"));
}

function addToDropdownMenu(suggestion_with_probability) {
  let index = suggestion_with_probability['index'];
  let original = suggestion_with_probability['original'];
  let trimmed = suggestion_with_probability['trimmed'];
  let probability = suggestion_with_probability['probability'];
  let source = suggestion_with_probability['source'];  // Could be empty

  // Hide empty string suggestions
  if (trimmed.length > 0) {
    $('#frontend-overlay').append(function() {
      return $('<div class="dropdown-item" data-source="' + source + '" data-original="' + original + '">' + trimmed + '</div>').click(function(){
        // console.log("click!");
        currentHoverIndex = index;
        currentIndex = index;
        selectDropdownItem(original);
      }).mouseover(function(){
        currentHoverIndex = index;
        logEvent(EventName.SUGGESTION_HOVER, EventSource.USER);
      }).data('index', index).data('original', original).data('trimmed', trimmed).data('probability', probability).data('source', source);
    });
  }

}

function reverse_sort_by_probability(a, b) {
  if (a.probability > b.probability ){
    return -1;
  }
  if (a.probability < b.probability){
    return 1;
  }
  return 0;
}

function addSuggestionsToDropdown(suggestions_with_probabilities) {
  emptyDropdownMenu();

  // Reverse sort suggestions based on probability if it is set in config
  if (sortSuggestions == true){
    suggestions_with_probabilities.sort(reverse_sort_by_probability);
  }

  for (let i = 0; i < suggestions_with_probabilities.length; i++) {
    addToDropdownMenu(suggestions_with_probabilities[i]);
  }

  items = $('.dropdown-item');
  numItems = items.length;
  currentIndex = 0;
}

function showDropdownMenu(source, is_reopen=false) {
  // Check if there are entries in the dropdown menu
  if ($('#frontend-overlay').children().length == 0) {
    if (is_reopen == true) {
        alert('You can only reopen suggestions when none of them was selected before. Please press tab key to get new suggestions instead!');
    } else {
        alert('No suggestions to be shown. Press tab key to get new suggestions!');
    }
    return;
  }
  else {
    // Compute offset
    let offsetTop = $('#editor-view').offset().top;
    let offsetLeft = $('#editor-view').offset().left;
    let offsetBottom = $('footer').offset().top;

    let position = quill.getBounds(getText().length);
    let top = offsetTop + position.top + 60 + 40;  // + Height of toolbar + line height
    let left = offsetLeft + position.left;

    // Fit frontend-overlay to the contents
    let maxWidth = 0;
    let totalHeight = 0;
    let finalWidth = 0;
    
    var editor_center = ($("#frontend").outerWidth(true) + parseInt($("#frontend").css("margin-left")))/2;
    var editor_max_width = Math.floor(($("#frontend").outerWidth(true) - parseInt($("#frontend").css("margin-left")))*0.8);
    $(".dropdown-item").each(function(){
        width = $(this).outerWidth(true);
        height = $(this).outerHeight(true);
        if (width > maxWidth) {
          maxWidth = width;
        }
        totalHeight = totalHeight + height;
    });


    if (width > editor_max_width){
      $(".dropdown-item").each(function(){
        $(this).width(editor_max_width);
      });
      
    }
    finalWidth = Math.min(maxWidth, editor_max_width);

    let rightmost = left + maxWidth;
    let bottommost = top + totalHeight;

    let width_overflow = rightmost > $("#editor-view").width();
    // Push it left if it goes outside of the frontend
    if (width_overflow) {
      left = offsetLeft + 30;  // 30px for padding
    }

    // Decide whether or not to move up the dropdown
    const bodyHeight = $('body').outerHeight(true);
    let moveUpDropdown = false;

    if (bottommost < ($("#editor-view").height() + 100)) {  // If it doesn't go over footer, no need to move up
    } else {  // If it does go over footer, then see whether moving up is easier
      if (top > (bodyHeight / 2)){
        moveUpDropdown = true;
      }
    }

    if (moveUpDropdown) {
      // console.log('$("#editor-view").height(): ' + $("#editor-view").height());
      // console.log('top: ' + top);
      // console.log('offsetTop: ' + offsetTop);
      // console.log('totalHeight: ' + totalHeight);

      // Adjust height
      var maxHeight = top - 100;
      if (totalHeight > maxHeight) {
        totalHeight = maxHeight;
      }

      // Set top
      top = top - totalHeight - 60;

    } else {
      // Set top
      top = top;

      // Adjust height
      var maxHeight = $("#editor-view").height() - offsetTop - position.top + 60;
      if (maxHeight < 100) {
        maxHeight = 100;
      }

      if (totalHeight > maxHeight){
        totalHeight = maxHeight;
      }

    }
    // alert($("#frontend").css("margin-left"));
    
    // $('#frontend-overlay-wrapper').css({
    //   top: bodyHeight - totalHeight - 100,
    //   left: bodyHeight;
    // });


    // Auto-select the first suggestion
    // if (domain != 'story') {
    //   $('#frontend-overlay > .dropdown-item').first().addClass('sudo-hover');
    // }
    $('#frontend-overlay').css({
      height: totalHeight,
    });

    

    
    // For sliding dropdown
    var x = 0,
      items = dropdown_container.find('.dropdown-item'),
      containerHeight = 0

    if(!dropdown_container.find('.dropdown-item:nth-child(2)').hasClass("sudo-hover")){
      dropdown_container.find('.dropdown-item:nth-child(2)').addClass("sudo-hover");
    }

    items.each(function(){
      if(x < numberVisibleDropdown){
        containerHeight = containerHeight + $(this).outerHeight();
        x++;
      }
    });
    dropdown_container.css({ height: containerHeight, overflow: "hidden" });
    suggestion_switch(dropdown_container.find('.dropdown-item:nth-child(2)'));
    // dropdown_container.find('.dropdown-item:nth-child(2)').click();
    
    // Set top and left
    $('#frontend-overlay').css({
      top: bodyHeight - containerHeight - 150,
      left: editor_center - finalWidth/2 - 20, // 20 is the left padding
    });
    openDropdownMenu(source, is_reopen);
  }
}

$('#ctrl-length_unit').on('change', function(e){
  // $('#ctrl-length_unit').attr('disabled', true);
  if (this.value == "none"){
    // $("#ctrl-length-slider-row").addClass("slideright");
    ctrl_len_slider.disabled(true);
    $('#ctrl-length_unit').parents(".card-body").removeClass('control');
  }
  else{
    $('#ctrl-length_unit').parents(".card-body").addClass('control');
    // $("#ctrl-length-slider-row").removeClass("slideright");
    // reset = reset_slider(value = [1,2], min = 1, max = 64);
    //  Have bug when setting min-max
    // if (this.value == "word"){
    //   reset = reset_slider(
    //     value = [10,25],
    //     // min = 1,
    //     // max = 64
    //     );
    // }
    ctrl_len_slider.disabled(false);
    // else if (this.value == "sentence"){
    //   reset = reset_slider(
    //     value = [1,2],
    //     min = 1,
    //     max = 8
    //     );
    // }
    // else if (this.value == "passage"){
    //   reset = reset_slider(
    //     value = [1,2],
    //     min = 1,
    //     max = 4
    //     );
    // }
  }
  // $('#ctrl-length_unit').attr('disabled', false);
});