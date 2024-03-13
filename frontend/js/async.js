async function startSession(accessCode) {
  session = {};
  logs = [];
  try {
    session = await wwai.api.startSession(domain, accessCode);
    if (debug) {
      console.log(session);
    }
    if (session['status'] == FAILURE){
      alert(session['message']);
    } else {
      sessionId = session.session_id;
      example = session.example;
      exampleActualText = session.example_text; 
      promptText = session.prompt_text;

      setPrompt(promptText);
      startTimer(session.session_length);
      promptLength = promptText.length;  // Number of characters
    
      if (isCounterEnabled == true) {
        resetCounter();
      }

      setCtrl(
        session.n,
        session.max_tokens,
        session.temperature,
        session.top_p,
        session.presence_penalty,
        session.frequency_penalty,
        // session.model_list,
      );

      stop = session.stop;
      engine = session.engine;
    }
  } catch(e) {
    alert('Start sesion error:' + e);
    console.log(e);
    session = await wwai.api.startSession(domain, accessCode);
    console.log(session)
  } finally {
    updateCounter();
    stopBlinking();

    if (domain == 'demo') {
      hideLoadingSignal();
    }
  }
}

async function endSession() {
  const results = await wwai.api.endSession(sessionId, logs);
  const verificationCode = results['verification_code'];

  $('#verification-code').removeClass('do-not-display');
  $('#verification-code').html('Verification code: ' + verificationCode);
}

async function endSessionWithReplay() {
  const results = await wwai.api.endSession(sessionId, logs);
  const verificationCode = results['verification_code'];
  // let replayLink = 'http://writingwithai-replay.glitch.me/?session_id=' + verificationCode;
  let replayLink = frontendURL + '/replay.html?session_id=' + verificationCode;

  if (domain == 'story') {
    // replayLink = 'http://writingwithai-replay-story.glitch.me/?session_id=' + verificationCode;
    replayLink = frontendURL + '/replay.html?session_id=' + verificationCode;
  }

  $('#verification-code').removeClass('do-not-display');
  $('#verification-code').html('<a href="' + replayLink + '" target="_blank">' + replayLink + '</a>');
}

const delay = ms => new Promise(res => setTimeout(res, ms));
var prevEventName = '';

async function replay(replayLogs, start){
  emptyDropdownMenu();
  quill.setText('');

  // Show query count
  $queryCurrent = $('#query-current');
  $queryTotal = $('#query-total');

  var queryCnt = 0;
  var queryTotal = 0;
  for (i = 0; i < replayLogs.length; i++) {
    if (replayLogs[i].eventName == EventName.SUGGESTION_GET) {
      queryTotal = queryTotal + 1;
    }
  }
  $queryCurrent.html(queryCnt);
  $queryTotal.html(queryTotal);

  // Show selection count
  $selectedCurrent = $('#selected-current');
  $selectedTotal = $('#selected-total');

  var selectedCnt = 0;
  var selectedTotal = 0;
  for (i = 0; i < replayLogs.length; i++) {
    if (replayLogs[i].eventName == EventName.SUGGESTION_SELECT) {
      selectedTotal = selectedTotal + 1;
    }
  }
  $selectedCurrent.html(selectedCnt);
  $selectedTotal.html(selectedTotal);

  // Show log count
  $logCurrent = $('#log-current');
  $logTotal = $('#log-total');

  var logCurrent = parseInt(start);

  $logCurrent.html(logCurrent);
  $logTotal.html(replayLogs.length + logCurrent - 1);

  // Show time elapse
  var timeOffset = replayLogs[0].eventTimestamp;
  var totalTime = (replayLogs[replayLogs.length - 1].eventTimestamp - timeOffset) / 1000;
  var elapsedTime = 0;

  var minutes = Math.floor(totalTime / 60).toString();
  var seconds = Math.floor(totalTime % 60).toString();
  if (minutes.length == 1) {
    minutes = '0' + minutes;
  }
  if (seconds.length == 1) {
    seconds = '0' + seconds;
  }

  $timeElapsed = $('#time-elapsed');
  $timeTotal = $('#time-total');

  $timeTotal.html(minutes + ':' + seconds);

  // If start is specified, initialize with currentDoc of the start log
  if (logCurrent > 0){
    setText(replayLogs[0].currentDoc);
    setCursor(replayLogs[0].currentCursor);
  }

  for (var i = 0; i < replayLogs.length; i++) {
    $logCurrent.html(logCurrent);
    logCurrent = logCurrent + 1;

    var logTime = replayLogs[i].eventTimestamp - timeOffset;
    timeOffset = replayLogs[i].eventTimestamp;

    elapsedTime += logTime;
    minutes = Math.floor(elapsedTime / 1000 / 60).toString();
    seconds = Math.floor(elapsedTime / 1000 % 60).toString();
    if (minutes.length == 1) {
      minutes = '0' + minutes;
    }
    if (seconds.length == 1) {
      seconds = '0' + seconds;
    }
    $timeElapsed.html(minutes + ':' + seconds);

    if (ReplayableEvents.includes(prevEventName)) {
      await delay(replayTime);
    }

    showLog(replayLogs[i]);

    // if (replayLogs[i].eventName == EventName.SYSTEM_INITIALIZE){
    //   await delay(10000); // NOTE Delete; for recording;
    // }

    // if (replayLogs[i].eventName == EventName.SUGGESTION_OPEN) {
    //   await delay(slowDownSuggestionTime);
    // }

    var prevEventName = replayLogs[i].eventName;
    var replayTime = Math.min(logTime / speedUpReplayTime, maximumElapsedTime);

    if (replayLogs[i].eventName == EventName.SUGGESTION_GET) {
      queryCnt += 1;
      $queryCurrent.html(queryCnt);
    }
    else if (replayLogs[i].eventName == EventName.SUGGESTION_SELECT) {
      selectedCnt += 1;
      $selectedCurrent.html(selectedCnt);
    }
  }

}

async function replayLogsWithSessionId(sessionId, range){
  try {
    results = await wwai.api.getLog(sessionId);

    let start = parseInt(range['start']);
    let end = parseInt(range['end']);
    if (end < 0){
      end = results['logs'].length;
    }
    replayLogs = results['logs'].slice(start, end + 1);

    await replay(replayLogs, start);

  } catch (e) {
    const message = 'We could not get the requested writing session (' + sessionId + ') '
                    + 'due to a server error or an invalid session ID.\n\n'
                    + 'Please notify the issue to ' + contactEmail + ' '
                    + 'and try again later.';
    setText(message);
    $('#editor-view').addClass('error');
    return;
  }
}

async function showFinalStoryWithSessionId(sessionId){
  console.log('Final mode: logs with session ID: ' + sessionId);
  try {
    results = await wwai.api.getLog(sessionId);
  } catch (e) {
    alert('Could not find logs for the given session ID: ' + sessionId + '\n\n' + e);
    return;
  }

  if (results['status'] == FAILURE) {
    alert(results['message']);
    return;
  }

  replayLogs = results['logs'];

  lastLog = replayLogs[replayLogs.length - 1];
  quill.setText(lastLog.currentDoc);
  quill.setSelection(lastLog.currentCursor);
}

async function loadLogsWithSessionId(newSessionId){
  try {
    results = await wwai.api.getLog(newSessionId);

    // Overwrite all settings with the ones from the logs
    config = results['config'];

    console.log('change sessionId from ' + sessionId + ' to ' + newSessionId);
    sessionId = newSessionId;
    example = config['example'];
    promptText = config['promptText'];
    if (promptText) {
      promptLength = promptText.length;
    } else {
      promptLength = 0;
    }

    setCtrl(
      config['n'],
      config['max_tokens'],
      config['temperature'],
      config['top_p'],
      config['presence_penalty'],
      config['frequency_penalty'],
      // config['model_list'],
    );
    stop = config['stop'];
    engine = config['engine'];

    // Overwrite the current logs with loaded logs
    loadedLogs = results['logs'];
    logs = loadedLogs;

    // Set the text editor to be the last state in the log
    const lastText = results['last_text'];
    const lastIndex = loadedLogs[results['logs'].length - 1]['currentCursor'];
    quill.setText(lastText, 'silent');
    quill.setSelection(lastIndex, 'silent');

    $('#editor-view-overlay').addClass('do-not-display');
    $('#start-btn').addClass('do-not-display');
    $('#load-btn').addClass('do-not-display');
    $('#save-btn').removeClass('do-not-display');
  } catch (e) {
    const message = 'We could not get the requested writing session (' + sessionId + ') '
                    + 'due to a server error or an invalid session ID.\n\n'
                    + e + '\n\n'
                    + 'Please notify the issue to ' + contactEmail + ' '
                    + 'and try again later.';
    alert(message);
    return;
  } finally {
    if (domain == 'reedsy') {
      hideLoadingSignal();
    }
  }
}

async function saveLogs() {
  try {
    results = await wwai.api.saveLog();
  } catch (e) {
    alert(e);
  } finally {
    if (domain == 'reedsy') {
      hideLoadingSignal();
      alert('Saved the current writing session!');
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Query
////////////////////////////////////////////////////////////////////////////////

function getDataForQuery(doc, exampleText, cursor_index, cursor_length, token_range_list = []) {
  // Check if the ctrl switch is checked. If not, don't send HMM control
  if (apply_control()){
    // n = $("#ctrl-n").val();
    // max_tokens = $("#ctrl-max_tokens").val();
    // temperature = $("#ctrl-temperature").val();
    // top_p = $("#ctrl-top_p").val();
    // presence_penalty = $("#ctrl-presence_penalty").val();
    // frequency_penalty = $("#ctrl-frequency_penalty").val();
    // model = $("#ctrl-model").val();
    keyword = $("#ctrl-keyword").val();
    length_unit = $("#ctrl-length_unit").val();
    length = [$('#ctrl-length-slider > input:eq(0)').val(), $('#ctrl-length-slider > input:eq(1)').val()];
    instruct = $("#ctrl-instruct").val();
    console.log("Send HMM control");
  }
  else {
    keyword = "";
    length_unit = "none";
    length = [];
    instruct = "";
    console.log("Skip HMM control");
  }
  const data = {
    'session_id': sessionId,
    'domain': domain,
    'example': example,
    'example_text': exampleText, // $('#exampleTextarea').val()

    'doc': doc,
    'cursor_index': cursor_index,
    'cursor_length': cursor_length,

    'n': $("#ctrl-n").val(),
    'max_tokens': $("#ctrl-max_tokens").val(),
    'temperature': $("#ctrl-temperature").val(),
    'top_p': $("#ctrl-top_p").val(),
    'presence_penalty': $("#ctrl-presence_penalty").val(),
    'frequency_penalty': $("#ctrl-frequency_penalty").val(),
    'stop': stop,
    // Model and Our Control Signal
    'model': $("#ctrl-model").val(),
    'keyword': keyword,
    'length_unit': length_unit,
    'length': length,
    'instruct': instruct,
    // By default, it is empty, which means that we do not do token length constraints
    'token_range_list': token_range_list,
    'engine': engine,

    'suggestions': getSuggestionState(),
  };
  // console.log("GetDataForQuery");
  return data;
}

function queryGPT3() {
  const doc = getText();
  // get the selected range
  let range = quill.getSelection();
  const exampleText = exampleActualText;
  const data = getDataForQuery(doc, exampleText, range.index, range.length);
  if (range.length > 0){
    // Store the to rewrite part
    update_to_rewrite(range.index, range.length);
  }
  else{
    reset_to_rewrite();
  }

  $.ajax({
    url: serverURL + '/api/query',
    beforeSend: function() {
      hideDropdownMenu(EventSource.API);
      // setCursorAtTheEnd();
      showLoadingSignal('Getting suggestions...');
    },
    type: 'POST',
    dataType: 'json',
    data: JSON.stringify(data),
    crossDomain: true,
    contentType: 'application/json; charset=utf-8',
    success: function(data) {
      hideLoadingSignal();
      if (data.status == SUCCESS) {
        if (data.original_suggestions.length > 0) {
          originalSuggestions = data.original_suggestions;
        } else {
          originalSuggestions = [];
        }

        if (data.suggestions_with_probabilities.length > 0) {
          let msg = 'Showing the suggestion stats!\n\n'
                    + '- could not think of suggestions (' + data.counts.empty_cnt + ')\n'
                    + '- generated same suggestions as before (' + data.counts.duplicate_cnt + ')\n'
                    + '- generated suggestions that contained banned words (' + data.counts.bad_cnt + ')\n';
          console.log(msg);
          addSuggestionsToDropdown(data.suggestions_with_probabilities);
          showDropdownMenu('api');
        } else {
          let msg = 'Please try again!\n\n'
                    + 'Why is this happening? The system\n'
                    + '- could not think of suggestions (' + data.counts.empty_cnt + ')\n'
                    + '- generated same suggestions as before (' + data.counts.duplicate_cnt + ')\n'
                    + '- generated suggestions that contained banned words (' + data.counts.bad_cnt + ')\n';
          console.log(msg);

          logEvent(EventName.SUGGESTION_FAIL, EventSource.API, textDelta=msg);
          alert("The system could not generate suggestions. Please try again.");
        }

      } else {
        alert(data.message);
      }
    },
    error: function() {
      hideLoadingSignal();
      alert("Could not get suggestions. Press tab key to try again! If the problem persists, please send a screenshot of this message to " + contactEmail + ". Our sincere apologies for the inconvenience!");
    }
  });
}

function get_token_range_list(){
  token_range_list = new Array();
  for (i = ctrl_token_range_min; i < ctrl_token_range_max; i+=ctrl_token_range_step) {
    token_range_list.push([i, i+ctrl_token_range_step]);
  }
  return token_range_list;
}

function queryTokenRange() {
  const doc = getText();
  // get the selected range
  let range = quill.getSelection();;
  const exampleText = exampleActualText;
  const data = getDataForQuery(doc, exampleText, range.index, range.length, token_range_list = get_token_range_list());
  // Here we add special 

  $.ajax({
    url: serverURL + '/api/query',
    // beforeSend: function() {
      // hideDropdownMenu(EventSource.API);
      // setCursorAtTheEnd();
      // showLoadingSignal('Getting suggestions...');
    // },
    type: 'POST',
    dataType: 'json',
    data: JSON.stringify(data),
    crossDomain: true,
    contentType: 'application/json; charset=utf-8',
    success: function(data) {
      if (data.status == SUCCESS) {
        if (data.original_suggestions.length > 0) {
          originalSuggestions = data.original_suggestions;
        } else {
          originalSuggestions = [];
        }

        if (data.suggestions_with_probabilities.length > 0) {
          let msg = 'Showing the suggestion stats!\n\n'
                    + '- could not think of suggestions (' + data.counts.empty_cnt + ')\n'
                    + '- generated same suggestions as before (' + data.counts.duplicate_cnt + ')\n'
                    + '- generated suggestions that contained banned words (' + data.counts.bad_cnt + ')\n';
          console.log(msg);
          // For token length control, don't need this. Instead, we need to do things
          // addSuggestionsToDropdown(data.suggestions_with_probabilities);
          // showDropdownMenu('api');
          update_token_output_list(data.original_suggestions);
          token_loaded_success();
        } else {
          let msg = 'Please try again!\n\n'
                    + 'Why is this happening? The system\n'
                    + '- could not think of suggestions (' + data.counts.empty_cnt + ')\n'
                    + '- generated same suggestions as before (' + data.counts.duplicate_cnt + ')\n'
                    + '- generated suggestions that contained banned words (' + data.counts.bad_cnt + ')\n';
          console.log(msg);

          logEvent(EventName.SUGGESTION_FAIL, EventSource.API, textDelta=msg);
          alert("The system could not generate suggestions. Please try again.");
          token_loaded_failed();
        }

      } else {
        alert(data.message);
        token_loaded_failed();
      }
    },
    error: function() {
      hideLoadingSignal();
      alert("Could not get token range suggestions! If the problem persists, please send a screenshot of this message to " + contactEmail + ". Our sincere apologies for the inconvenience!");
      token_loaded_failed();
    }
  });
}
