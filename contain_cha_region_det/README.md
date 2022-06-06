* container_cha_region_det文件夹中包含所有的集装箱字符识别数据的处理代码
* step1:clip_charcter_region_for_ocr_cha_det.py:
        1）并挑选出有多个目标检测区域的图片，保存在文件夹中，不做任何处理
        2）对非多目标检测区域的图片做以下处理：
          得出做目标检测区域的标签
          裁剪出目标检测标签框选出的区域
          得出用于文字区域分割的标签
* step2:draw_rectangle_on_img.py:将文字区域的外接矩形框绘制在图片上,便于人工检查
* step3:make_det_label.py:将detbox转换为可以用于yolov5训练的标签
* ------20210507更新-------------*


----
20211019更新<br>
原因：数据组做标签的方法改变了，有了集装箱号的外接框
----
* 更新后标签制作过程：<br>
  step1：read_label_from_cvat_for_cha_rec_new.py 从CVAT中读出所有标签;<br>
  step2：make_large_region_labels.py 制作大块区域的标签，用于目标检测;<br>
  step3：clip_charcter_region_for_ocr_cha_det_new.py 制作字符识别（目标检测）的标签;<br>
  step4：draw_rectangle_on_img_new.py 将制作的标签（符合yolov5训练标准）绘制在图片上;<br>
  step5: split_train_test_imgs.py 将整理好的数据划分为训练样本和测试样本<br>
  step6: copy_img_for_train_test.py将step5中的样本拷贝，进行模型训练<br>
  step7: sum_img_label.py统计各个类别的字符数量
* ----------20211019更新-------------*
