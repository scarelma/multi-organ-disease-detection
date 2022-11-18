<?php
$file_name =$_FILES['file']['name'];
$temp_name =$_FILES['file']['tmp_name'];
$file_up_name= time().$file_name;
move_uploaded_file($temp_name, "files/".$file_up_name);

?>
