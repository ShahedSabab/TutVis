

$(document).ready(function() {
  window.localStorage.clear();
  window.localStorage.setItem('topic', "");
  window.localStorage.setItem('difficulty', "");
  var header = document.getElementById("myDIV");
  var btns = header.getElementsByClassName("btn");
  for (var i = 0; i < btns.length; i++) {
    btns[i].addEventListener("click", function() {
      var current = document.getElementsByClassName("active");
      if (current.length > 0) {
        current[0].className = current[0].className.replace(" active", "");
      }
      this.className += " active";
      window.localStorage.setItem('topic', this.innerText);
    });
  }
  var header2 = document.getElementById("myDIV2");
  var btns2 = header2.getElementsByClassName("btn");
  for (var i = 0; i < btns2.length; i++) {
    btns2[i].addEventListener("click", function() {
      var current2 = header2.getElementsByClassName("active");
      if (current2.length > 0) {
        current2[0].className = current2[0].className.replace(" active", "");
      }
      this.className += " active";
      window.localStorage.setItem('difficulty', this.innerText);
    });
  }

  var search = document.getElementById("search-homepage");
  search.addEventListener("click", function() {
    var topic = localStorage.getItem('topic');
    var difficulty = localStorage.getItem('difficulty');
    if(topic.length==0 && difficulty.length==0)
    {
      alert("Select at least a topic or difficulty");
    }
    else
    {
      document.getElementById("homepage").style.visibility="hidden";
    }

  });




  $('a').click(function(event) {
        var currentElemID = $(this).attr("id");
        var currentElemType = $(this).attr("type");
        loadIframe("inlineFrameExample", currentElemID);
    });
});



function loadIframe(iframeName, id) {

    url = "tutorial/"+id.toString()+".html";
    var $iframe = $('#' + iframeName);
    if ( $iframe.length ) {
        $iframe.attr('src',url);
        return false;
    }
    return true;
}







