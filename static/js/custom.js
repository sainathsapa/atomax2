// File Choosing trigger
$('#fileinputTrigger').click(function () {
    $('#fileinput').trigger('click');

}


);


// Default Color values
$('#sublink').hide();
$('#icon-file').addClass('color-a');
$('#icon-news').addClass('color-b');


// Function to handle the clicked events
$('#linkchoose').click(function () {
    $('#filechoose').removeClass('summary-grid-active');
    $('#linkchoose').addClass('summary-grid-active');
    $('#subfile').hide();

    $('#sublink').show();
    $('#icon-file').replaceClass('color-a', 'color-b');
    $('#icon-news').replaceClass('color-b', 'color-a');

});

$('#filechoose').click(function () {
    $('#filechoose').addClass('summary-grid-active');
    $('#linkchoose').removeClass('summary-grid-active');
    $('#subfile').show();

    $('#sublink').hide();
    $('#icon-file').replaceClass('color-b', 'color-a');
    $('#icon-news').replaceClass('color-a', 'color-b');




});




// Function for Replacement of Class Definition
(function ($) {
    $.fn.replaceClass = function (pFromClass, pToClass) {
        return this.removeClass(pFromClass).addClass(pToClass);
    };
}(jQuery));