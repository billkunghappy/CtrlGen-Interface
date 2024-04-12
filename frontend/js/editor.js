var checkFormatLockTime = new Date();  // for template

/* Setup */
function trackTextChanges(){
  quill.on('text-change', function(delta, oldDelta, source) {
    console.log("Text Change");
    if (source != 'user') {
      // We only track user interaction here
      return;
    }
    else {
      // Classify whether it's insert or delete
      eventName = null;
      eventSource = sourceToEventSource(source);

      ops = new Array();
      for (let i = 0; i < delta.ops.length; i++) {
        ops = ops.concat(Object.keys(delta.ops[i]));
      }
      if (ops.includes('insert')) {
        eventName = EventName.TEXT_INSERT;
      } else if (ops.includes('delete')) {
        eventName = EventName.TEXT_DELETE;
      } else {
        eventName = EventName.SKIP;
        console.log('Ignore format change');
      }
      logEvent(eventName, eventSource, textDelta=delta);

      if (isCounterEnabled == true) {
        updateCounter();
      }

      if (domain == 'template') {
        let currentTime = new Date();
        let elapsedTime = (currentTime - checkFormatLockTime) / 1000;

        if (elapsedTime > 1){
          checkFormatLockTime = currentTime;
          formatNonTerminals();
        }
      }
    }
  });
}

function trackTextChangesByMachineOnly(){
  quill.on('text-change', function(delta, oldDelta, source) {
    eventName = null;
    eventSource = sourceToEventSource(source);

    ops = new Array();
    for (let i = 0; i < delta.ops.length; i++) {
      ops = ops.concat(Object.keys(delta.ops[i]));
    }
    if (ops.includes('insert')) {
      eventName = EventName.TEXT_INSERT;
    } else if (ops.includes('delete')) {
      eventName = EventName.TEXT_DELETE;
    } else {
      eventName = EventName.SKIP;
      console.log('Ignore format change');
    }

    // Ignore text-change by user and reset to oldDelta
    if (source == 'silent') {
      return;
    } else if (source == EventSource.API) {
      logEvent(eventName, eventSource, textDelta=delta);
    // Allow deletion
    } else if (source == EventSource.USER && eventName == EventName.TEXT_DELETE) {
      logEvent(eventName, eventSource, textDelta=delta);
    // Allow insertion of whitespace
    } else if (source == EventSource.USER && eventName == EventName.TEXT_INSERT){
        const isInsert = (element) => element == 'insert';
        let index = ops.findIndex(isInsert);

        if (delta.ops[index]['insert'].trim() == '') {
          logEvent(eventName, eventSource, textDelta=delta);
        } else {
          quill.setContents(oldDelta, 'silent');
        }
    } else {
      console.log('Ignore unknown change:', source, eventName);
    }

    if (isCounterEnabled == true) {
      updateCounter();
    }

  });
}

function trackSelectionChange(){
  // NOTE It's "silenced" when coincide with text-change
  quill.on('selection-change', function(range, oldRange, source) {
    if (range === null) {
      return;  // click outside of the editor
    } else if (source == 'silent'){
      return;
    } else {
      eventName = null;
      eventSource = sourceToEventSource(source);

      // Use prevCursorIndex instead of oldRange.index as oldRange is null at times
      if (range.length > 0){
        eventName = EventName.CURSOR_SELECT;
      } else if (range.index > prevCursorIndex) {
        eventName = EventName.CURSOR_FORWARD;
      } else if (range.index < prevCursorIndex) {
        eventName = EventName.CURSOR_BACKWARD;
      } else if (range.index == prevCursorIndex){
        // Deselect
        eventName = EventName.SKIP;
      } else {
        if (debug) {
          alert('Wrong selection-change handling!');
          console.log(range, oldRange, source);
        }
        eventName = EventName.SKIP;
      }

      logEvent(eventName, eventSource, textDelta='', cursorRange=range);
    }
  });
}

function setupEditorHumanOnly() {
  quill = new Quill('#editor-container', {
    theme: 'snow',
    placeholder: 'Write something...',
    modules: {
      clipboard: {
        matchVisual: false,  // Prevent empty paragraph to be added
        matchers: [
						[
              Node.ELEMENT_NODE, function(node, delta) {
  							return delta.compose(new Delta().retain(delta.length(), {
                    color: false,
                    background: false,
                    bold: false,
                    strike: false,
                    underline: false
  							}));
						  }
						]
					]
      },
    }
  });

  trackTextChanges();
  trackSelectionChange();

  quill.focus();
}

function setupEditorMachineOnly() {
  let bindings = {
    tab: {
      key: 9,
      handler: function() {
        logEvent(EventName.SUGGESTION_GET, EventSource.USER);
        queryGPT3();
        if (apply_control() && apply_token_control){
          queryTokenRange();
        }
      }
    },
    enter: {
      key: 13,
      handler: function() {
        let selectedItem = $('.sudo-hover');
        if (selectedItem.length > 0) {
          $(selectedItem).click();
        } else {
          return true;
        }
      }
    }
  };

  quill = new Quill('#editor-container', {
    theme: 'snow',
    modules: {
      keyboard: {
        bindings: bindings
      },
      clipboard: {
        matchVisual: false,  // Prevent empty paragraph to be added
        matchers: [
						[
              Node.ELEMENT_NODE, function(node, delta) {
  							return delta.compose(new Delta().retain(delta.length(), {
                    color: false,
                    background: false,
                    bold: false,
                    strike: false,
                    underline: false
  							}));
						  }
						]
					]
      },
    }
  });

  trackTextChangesByMachineOnly();
  trackSelectionChange();

  quill.focus();
}

function setupEditor() {
  let bindings = {
    tab: {
      key: 9,
      handler: function() {
        logEvent(EventName.SUGGESTION_GET, EventSource.USER);
        queryGPT3();
        if (apply_control() && apply_token_control){
          queryTokenRange();
        }
      }
    },
    enter: {
      key: 13,
      handler: function() {
        let selectedItem = $('.sudo-hover');
        if (selectedItem.length > 0) {
          $(selectedItem).click();
        } else {
          return true;
        }
      }
    }
  };

  quill = new Quill('#editor-container', {
    theme: 'snow',
    modules: {
      keyboard: {
        bindings: bindings
      },
      clipboard: {
        matchVisual: false,  // Prevent empty paragraph to be added
        matchers: [
						[
              Node.ELEMENT_NODE, function(node, delta) {
  							return delta.compose(new Delta().retain(delta.length(), {
                    color: false,
                    background: false,
                    bold: false,
                    strike: false,
                    underline: false
  							}));
						  }
						]
					]
      },
    }
  });

  trackTextChanges();
  trackSelectionChange();

  quill.focus();
}

/* Cursor */
function getCursorIndex() {
  let range = quill.getSelection();

  if (range) {
    if (range.length == 0) {
      prevCursorIndex = range.index;
      return range.index;
    } else {
      // For selection, return index of beginning of selection
        prevCursorIndex = range.index;
      return range.index; // Selection
    }
  } else {
    return prevCursorIndex; // Not in editor
  }
}

function setCursor(index) {
  console.log("set cursor");
  // Adjust if index is outside of range
  let doc = quill.getText(0);
  let lastIndex = doc.length - 1;
  if (lastIndex < index) {
    index = lastIndex;
  }

  quill.setSelection(index);
  prevCursorIndex = index;
}

function setCursorAtTheEnd() {
  // If it's not triggerd by user's text insertion, and instead by api's
  // forced selection change, then it is saved as part of logs by selection-change
  let doc = quill.getText(0);
  let index = doc.length - 1; // The end of doc
  setCursor(index);
}

/* Text */
function getText() {
  let text = quill.getText(0);
  return text.substring(0, text.length - 1);  // Exclude trailing \n
}

function setText(text) {
  // NOTE Does not keep the formating
  quill.setText(text, 'api');
  setCursor(text.length);
}

function appendText(text) {
  // If there's text to rewrite, we first remove it
  remove_to_rewrite();
  // Now we can insert the new text, and also store these machine generated text as to_rewrite
  let curIndex = getCursorIndex();
  console.log("Insert text: #" + text + "# -- "+ String(curIndex));
  quill.insertText(curIndex, text, source = "api");
  update_to_rewrite(curIndex, text.length);
  setCursor(curIndex + text.length);
  // Show the token control panel
  show_token_control();

  // Start a one time text change caller. If text been changed by user (only user), remove all format and disable token control.
  // Also after text changed, since we disable the to rewrite, we also need to 
  // For token length control, we will call it with silent
  quill.once('text-change', (delta, oldDelta, source) => {
    if (source == "user") {
      // Note: If the text is changed by calling append text
      reset_to_rewrite();
      remove_all_format();
      disable_token_control();
      console.log("Editor changed by user, remove format and disable token control...")
    }
  });
}


function remove_all_format(){
  quill.formatText(0, 100000, {
    'color': 'black'
  });
  console.log("remove format");
}


// We have a function to store the to_rewrite part
var to_rewrite_curIndex = -1;
var to_rewrite_length = -1;


function reset_to_rewrite(){
  to_rewrite_curIndex = -1;
  to_rewrite_length = -1;
}

function update_to_rewrite(curIndex, length){
  // Update the information of the to_rewrite span. In the furture, when remove_to_rewrite is called, we rewrite it
  to_rewrite_curIndex = curIndex;
  to_rewrite_length = length;
  // console.log("Set to_rewrite: " + String(to_rewrite_curIndex) + ", " + String(to_rewrite_length));
  // Update format
  remove_all_format();
  // Add color to the inserted text
  quill.formatText(curIndex, length, {
    'bold' : false,
    // 'color': 'rgb(128,128,128)'
    'color': 'rgb(118, 171, 174)'
  });
  // console.log("set format at " + String(curIndex) + " ~ " + String(curIndex + length));
}

function remove_to_rewrite(){
  // Will remove the rewritten part if they are set. And will remove all format. After that, reset the to_rewrite vars
  if (to_rewrite_curIndex >= 0 && to_rewrite_length > 0){
    quill.deleteText(to_rewrite_curIndex, to_rewrite_length);
    setCursor(to_rewrite_curIndex);
    remove_all_format();
    console.log("Delete to_rewrite: " + String(to_rewrite_curIndex) + ", " + String(to_rewrite_length));
  }
  reset_to_rewrite();
}