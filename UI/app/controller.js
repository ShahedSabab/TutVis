/**
 * Main application controller
 *
 * You can use this controller for your whole app if it is small
 * or you can have separate controllers for each logical section
 *
 */
;(function () {

  angular
    .module('boilerplate')
    .controller('MainController', MainController);


  MainController.$inject = ['LocalStorage', 'QueryService', '$scope', '$http'];


  function MainController(LocalStorage, QueryService, $scope, $http) {


    $scope.$on('$locationChangeStart', function(event, next, current){
      event.preventDefault();
    });

    console.log("calling");
    // var path = "./images/testing.csv";
    var path = "./images/data2.csv";
    $scope.database = [];
    $scope.records = [];
    $scope.flag = 0;
    $scope.foundResult = "Tutorial List";
    $http.get(path)
      .success(function (data, status, headers, config) {
        $scope.flag = 1;
        $scope.database = csvJSON(data);
        console.log($scope.database);
        for(var i=0; i<$scope.database.length; i++){
         var resultTopics = $scope.database[i].topic;
         var resultTools = $scope.database[i].tools;
         var resultTopicCon = $scope.database[i].topic_con;
         var changeTopic = resultTopics.replace(/\#/g, ", ");
         var res = resultTopics.replace(/\#/g, ",");
         var resTools = resultTools.replace(/\#/g, ",");
         var resTools = resTools.replace(/ /g, "-");
          console.log("contribution");
         console.log(resultTopicCon);
         var resTopicCon = resultTopicCon.replace(/\#/g, ",");

         $scope.database[i].topic = changeTopic;
         var topic = res.split(",");
         var tools = resTools.split(",");
         var con = resTopicCon.split(",");

         //calculating the percentage of the contribution of the topics
         var totalCon =sum(con);

         for (var j=0; j<con.length;j++)
         {
           con[j] = con[j]*100/totalCon;
         }

         for (var j=0; j<topic.length;j++)
         {
           if(topic[j]=="Photo Editing Manipulation & Special Effects")
           {
             topic[j]= "Photo Editing, Manipulation & Special Effects";
           }
           else if(topic[j]=="Shading Texture & Color Blending")
           {
             topic[j]="Shading, Texture & Color Blending";
           }
           else if(topic[j]=="File Organization Share & Export")
           {
             topic[j]="File Organization, Share & Export";
           }
         }

         $scope.database[i].topicList = topic;
         $scope.database[i].toolsList = tools;
         $scope.database[i].topicConList = con;
        }
        console.log($scope.database);
        $scope.records = $scope.database;
      })
      .error(function (data, status, header, config) {

      })


    function sum( obj ) {
      var sum = 0;
      for( var el in obj ) {
        if( obj.hasOwnProperty( el ) ) {
          sum += parseFloat( obj[el] );
        }
      }
      return sum;
    }



    //apply filter and searching
    $scope.myLoadListFunc = function () {


      var checkedList = [];
      var checkedDifficultyList = [];
      $.each($("input[name='topicList']:checked"), function () {
        checkedList.push($(this).parent().find("label").text());
      });
      $.each($("input[name='difficultyList']:checked"), function () {
        checkedDifficultyList.push($(this).parent().find("label").text());
      });

      if (checkedList.length == 0 && checkedDifficultyList.length == 0) {
        jQuery.noConflict();
        $("#myModal").modal('show');
      } else if (checkedList.length > 0 && checkedDifficultyList.length > 0) {
        $("#sidebar-wrapper").hide();
        $("#loadingTutorials").show();
        $scope.records = [];
        $scope.j = 0;
        for (var i = 0; i < checkedList.length; i++) {
          var topicName = checkedList[i];
          for (var j = 0; j < checkedDifficultyList.length; j++) {
            var difficulty = checkedDifficultyList[j];
            loadList(topicName, difficulty, $scope, "T1D1");
          }
        }
        setTimeout(function () {
          $("#loadingTutorials").hide();
          $("#sidebar-wrapper").fadeIn()
            .animate({top: 275}, 800, function () {
              //callback
            });
        }, 1000);

        if ($scope.j <= 1) {
          $scope.foundResult = "Tutorial List (" + $scope.j + " result found ...)";
        } else {
          $scope.foundResult = "Tutorial List (" + $scope.j + " results found ...)";
        }
      } else if (checkedList.length > 0) {
        $("#sidebar-wrapper").hide();
        $("#loadingTutorials").show();
        $scope.records = [];
        $scope.j = 0;
        for (var i = 0; i < checkedList.length; i++) {
          var topicName = checkedList[i];
          console.log("next");
          loadList(topicName, difficulty, $scope, "T1D0");
        }
        setTimeout(function () {
          $("#loadingTutorials").hide();
          $("#sidebar-wrapper").fadeIn()
            .animate({top: 275}, 800, function () {
              //callback
            });
        }, 1000);
        if ($scope.j <= 1) {
          $scope.foundResult = "Tutorial List (" + $scope.j + " result found ...)";
        } else {
          $scope.foundResult = "Tutorial List (" + $scope.j + " results found ...)";
        }
      } else if (checkedDifficultyList.length > 0) {
        $("#sidebar-wrapper").hide();
        $("#loadingTutorials").show();
        $scope.records = [];
        $scope.j = 0;
        for (var i = 0; i < checkedDifficultyList.length; i++) {
          var difficulty = checkedDifficultyList[i];
          loadList(topicName, difficulty, $scope, "T0D1");
        }
        setTimeout(function () {
          $("#loadingTutorials").hide();
          $("#sidebar-wrapper").fadeIn()
            .animate({top: 275}, 800, function () {
              //callback
            });
        }, 1000);

        console.log("records"+$scope.records);

        if ($scope.j <= 1) {
          $scope.foundResult = "Tutorial List (" + $scope.j + " result found ...)";
        } else {
          $scope.foundResult = "Tutorial List (" + $scope.j + " results found ...)";
        }
      }
    };


    $scope.myLinkLoadFunc = function (id) {
      console.log(id);
      $("#searchText").hide();
      $("#wrapper").toggleClass("toggled");
      $("#page-content-wrapper").width("100%");
      $("#main").hide("slow");
      $("#menu-toggle").show();
      loadIframe("inlineFrameExample", id);
      $("a").css('background-color', '');
      $("#" + id.toString()).css("background-color", "silver");
      //change if you want to resize the selector of the list item
      $("#" + id.toString()).css("border-right", "26vh solid #FFFFFF");

    };

    //menu toggle to came back to the list from tutorial
    $scope.myMenuToggleFunc = function () {
      $("#wrapper").toggleClass("toggled");
      $("#page-content-wrapper").width("0px");
      $("#main").show("slow");
      $("#menu-toggle").hide();
      $("#searchText").show();
      loadIframe("inlineFrameExample", 'topic0');
    }



    // mouse hover start
    $scope.spanHover = function (id, hoverId, graphicId) {
      $('#spanTopic'+hoverId.toString()+graphicId.toString()+id.toString()).css("opacity", "0.5");
      $('#graphicTopic'+hoverId.toString()+graphicId.toString()+id.toString()).css("opacity", "0.5");
    }
    $scope.spanNonHover = function (id, hoverId, graphicId) {
      $('#spanTopic'+hoverId.toString()+graphicId.toString()+id.toString()).css("opacity", "1");
      $('#graphicTopic'+hoverId.toString()+graphicId.toString()+id.toString()).css("opacity", "1");
    }
    $scope.graphicHover = function (id, hoverId, graphicId) {
      $('#graphicTopic'+hoverId.toString()+graphicId.toString()+id.toString()).css("opacity", "0.5");
    }
    $scope.graphicNonHover = function (id, hoverId, graphicId) {
      $('#graphicTopic'+hoverId.toString()+graphicId.toString()+id.toString()).css("opacity", "1");
    }

    // mouse hover end

    $(document).ready(function () {

      // new Chart(document.getElementById("doughnut-chart"), {
      //   type: 'doughnut',
      //   data: {
      //     labels: ["Africa", "Asia", "Europe", "Latin America", "North America"],
      //     datasets: [
      //       {
      //         label: "Population (millions)",
      //         backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850"],
      //         data: [50,20,10,15,5]
      //       }
      //     ]
      //   },
      //   options: {
      //     tooltip:{enabled:true, bodyFontSize:20, titleFontSize:20},
      //     legend: { display: false },
      //     title: {
      //       display:false,
      //       text: 'Predicted world population (millions) in 2050'
      //     }
      //   }
      // });
      // $('a').click(function(event) {
      //   alert("clicked");
      //   var currentElemID = $(this).attr("id");
      //   var currentElemType = $(this).attr("type");
      //   //loadIframe("inlineFrameExample", currentElemID);
      //   console.log("clicked ====== "+currentElemID.toString());
      //   $("a").css('background-color', '');
      //   $("#"+currentElemID.toString()).css("background-color","silver");
      //   $("#"+currentElemID.toString()).css("border-right","30vh solid #FFFFFF");
      // });

      // $('div.pie-chart').each(function(i, d) {
      //
      //
      //   alert(d.id.toString());
      //   console.log(d.id.toString());
      //   var data = [
      //     {x: "Sketching & Digital Painting", normal: {fill: "#44AA99"},value: 0.0372, words:"color, digital_art, rough_sketch, motif, focal_point"},
      //     {x: "MIX: Editing and Transformation", normal: {fill: "#DDCC77"},value: 0.1038, words:"merge, duplicate, add, fill, warp"},
      //     {x: "Drawing Pixel Art",normal: {fill: "#CC6677"}, value: 0.6563, words:"pixel_art, pencil, draw, isometric, volume"}
      //   ];
      //   var chart = anychart.pie(data);
      //   /* set the inner radius
      //   (to turn the pie chart into a doughnut chart)*/
      //   chart.innerRadius("30%");
      //
      //   var animationSettings = chart.animation();
      //   animationSettings.duration(500);
      //   animationSettings.enabled(true);
      //   var tips = chart.tooltip();
      //   tips.format('Words related to this topic: \n {%words}');
      //   tips.title().fontSize(12);
      //   tips.fontSize(11);
      //   tips.width(172);
      //   tips.height(140);
      //   //tips.useHtml(true);
      //   //tips.title("<span style='font-size:12px; font-style:italic'>" );
      //   chart.hovered().explode("5%");
      //   chart.legend(false);
      //   // set the container id
      //   chart.container(d.id.toString());
      //   console.log(d.id.toString());
      //   chart.labels().enabled(false);
      //   chart.background().fill('#c1ffc1 0');
      //
      //   // initiate drawing the chart
      //   chart.draw();
      // });

      // for(var i =1;i<2;i++)
      // {
      //   var data = [
      //     {x: "Photo Editing, Manipulation & Special Effects", normal: {fill: "#44AA99"},value: 14, words:"thumbnail, manipulation, brightness_contrast, man_portrait, threshold_level"},
      //     {x: "Sketching and Artwork", normal: {fill: "#DDCC77"},value: 22, words:"sketch, art, work"},
      //     {x: "Drawing Pixel Art",normal: {fill: "#CC6677"}, value: 40, words:"draw, isometric, charge"}
      //   ];
      //   var chart = anychart.pie(data);
      //   /* set the inner radius
      //   (to turn the pie chart into a doughnut chart)*/
      //   chart.innerRadius("30%");
      //
      //   var animationSettings = chart.animation();
      //   animationSettings.duration(500);
      //   animationSettings.enabled(true);
      //   var tips = chart.tooltip();
      //   tips.format('Words related to this topic: \n {%words}');
      //   tips.title().fontSize(12);
      //   tips.fontSize(11);
      //   tips.width(172);
      //   tips.height(140);
      //   //tips.useHtml(true);
      //   //tips.title("<span style='font-size:12px; font-style:italic'>" );
      //   chart.hovered().explode("5%");
      //   chart.legend(false);
      //   // set the container id
      //   chart.container("pie"+i.toString());
      //   chart.labels().enabled(false);
      //   chart.background().fill('#c1ffc1 0');
      //
      //   // initiate drawing the chart
      //   chart.draw();
      // }


    });

    ////////////  function definitions

    // this.x = 1115;

    /**
     * Load some data
     * @return {Object} Returned object
     */
    // QueryService.query('GET', 'posts', {}, {})
    //   .then(function(ovocie) {
    //     self.ovocie = ovocie.data;
    //   });
  }


  function csvJSON(csv) {

    var lines = csv.split("\n");

    var result = [];

    var headers = lines[0].split(",");

    for (var i = 1; i < lines.length; i++) {

      var obj = {};
      var currentline = lines[i].split(",");

      for (var j = 0; j < headers.length; j++) {
        obj[headers[j]] = currentline[j];
      }

      result.push(obj);

    }
    result.splice(-1, 1);
    return result; //JavaScript object
    //return JSON.stringify(result); //JSON
  }


  function loadIframe(iframeName, id) {

    url = "tutorial/" + id.toString() + ".html";
    var $iframe = $('#' + iframeName);
    if ($iframe.length) {
      $iframe.attr('src', url);
      return false;
    }
    return true;
  }

  function loadList(topicName, difficulty, $scope, check) {

    $scope.flag = 0;
    if (check == "T1D0") {
      var z = 0;
      while(z<5) {
        for (var i = 0; i < $scope.database.length; i++) {
          if(z<$scope.database[i].topicList.length){
            if ($scope.database[i].topicList[z] == topicName) {
              var flag=0;
              for(var k=0; k<$scope.records.length;k++)
              {
                if($scope.database[i]==$scope.records[k])
                {
                  flag=1;
                }
              }
              if(flag==0)
              {
                $scope.records[$scope.j] = $scope.database[i];
                console.log($scope.j);
                console.log($scope.records[$scope.j]);
                $scope.j++;
              }
            }
          }
        }
        z++;
      }
    } else if (check == "T0D1") {
      for (var i = 0; i < $scope.database.length; i++) {
        if ($scope.database[i].label == difficulty) {
          $scope.records[$scope.j] = $scope.database[i];
          console.log($scope.j);
          console.log($scope.records[$scope.j]);
          $scope.j++;
        }
      }
    } else if (check == "T1D1") {
      var z = 0;
      while(z<5) {
        for (var i = 0; i < $scope.database.length; i++) {
          if(z<$scope.database[i].topicList.length) {
            if ($scope.database[i].label == difficulty && $scope.database[i].topicList[z] == topicName) {
              var flag = 0;
              for(var k=0; k<$scope.records.length;k++)
              {
                if($scope.database[i]==$scope.records[k])
                {
                  flag=1;
                }
              }
              if(flag==0) {
                $scope.records[$scope.j] = $scope.database[i];
                console.log($scope.j);
                console.log($scope.records[$scope.j]);
                $scope.j++;
              }
            }
          }
        }
        z++;
      }

    }
    $scope.flag = 1;
  }




})();
