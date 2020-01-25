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
    .controller('ChartController', ChartController);


  ChartController.$inject = ['LocalStorage', 'QueryService', '$scope', '$http'];


  function ChartController(LocalStorage, QueryService, $scope, $http) {



    $scope.init = function (id) {
      console.log("chart controller init "+id+ " database length "+ $scope.database.length);


      for (var i=0; i<$scope.database.length; i++)
      {

        if($scope.database[i].ID==id)
        {
          $('.value').each(function () {
            var text = $(this).text();
            $(this).parent().css('width', text);
          });
          $('.block').tooltip();
        }

      }
    };
  }
  })();
