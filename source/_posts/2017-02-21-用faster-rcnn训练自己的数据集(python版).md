---
layout: post
title: "用faster-rcnn训练自己的数据集(VOC2007格式,python版)"
date: 2017-02-21 11:24:00
description: "用faster-rcnn训练自己的数据集"
category: [deep learning]
tags: [python]
---

用faster-rcnn训练自己的数据集(VOC2007格式,python版)

<!--more-->

## 一. 配置caffe环境

[ubunt16.04下caffe环境安装](http://report.opsauto.cn/deep%20learning/2016/11/12/ubunt16.04%E4%B8%8Bcaffe%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85.html)

## 二. 下载,编译及测试py-faster-rcnn源码

### (一)下载源码

[github链接](https://github.com/rbgirshick/py-faster-rcnn)

或者执行 git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git

注意加上--recursive关键字

### (二)编译源码

编译过程中可能会出现缺失一些python模块,按提示安装

#### (1)编译Cython模块

	cd $FRCN_ROOT/lib 
	make

#### (2)修改Markfile配置
参考[ubunt16.04下caffe环境安装](http://report.opsauto.cn/deep%20learning/2016/11/12/ubunt16.04%E4%B8%8Bcaffe%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85.html)
中修改Makefile.config

#### (3)编译python接口

	cd $FRCN_ROOT/caffe-fast-rcnn
	make -j8  多核编译,时间较长
	make pycaffe

#### (4)下载训练好的VGG16和ZF模型

	cd $FRCN_ROOT
	./data/scripts/fetch_faster_rcnn_models.sh

时间太长的话可以考虑找网上别人分享的资源

### (三)测试源码

	cd $FRCN_ROOT
	./tool/demo.py

## 三. 使用faster-rcnn训练自己的数据集

### (一)下载预训练参数及模型

	cd $FRCN_ROOT
	./data/scripts/fetch_imagenet_models.sh
	./data/scripts/fetch_selective_search_data.sh

### (二)制作数据集

[制作数据集(VOC2007格式)](http://report.opsauto.cn/deep%20learning/2017/02/08/%E5%88%B6%E4%BD%9C%E8%87%AA%E5%B7%B1%E7%9A%84%E5%9B%BE%E7%89%87%E6%95%B0%E6%8D%AE%E9%9B%86.html)

将制作好的VOC2007文件夹放置在data/VOCdevkit2007文件夹下,没有则新建VOCdevkit2007文件夹

### (三)修改配置文件

#### (1)修改py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_fast_rcnn_train.pt和stage2_fast_rcnn_train.pt 两个文件

备注:3处修改及其附近的代码
	
	name: "ZF"
	layer {
	  name: 'data'
	  type: 'Python'
 	 top: 'data'
  	top: 'rois'
  	top: 'labels'
 	 top: 'bbox_targets'
 	 top: 'bbox_inside_weights'
  	top: 'bbox_outside_weights'
  	python_param {
  	  module: 'roi_data_layer.layer'
  	  layer: 'RoIDataLayer'
  	  param_str: "'num_classes': 2" #按训练集类别改，该值为类别数+1
  	}
	}

	layer {
 	 name: "cls_score"
 	 type: "InnerProduct"
 	 bottom: "fc7"
 	 top: "cls_score"
 	 param { lr_mult: 1.0 }
 	 param { lr_mult: 2.0 }
  	inner_product_param {
    	num_output: 2 #按训练集类别改，该值为类别数+1
   	 weight_filler {
   	   type: "gaussian"
   	   std: 0.01
   	 }
   	 bias_filler {
  	    type: "constant"
   	   value: 0
  	  }
 	 }
	}

	layer {
	  name: "bbox_pred"
 	 type: "InnerProduct"
 	 bottom: "fc7"
  	top: "bbox_pred"
 	 param { lr_mult: 1.0 }
 	 param { lr_mult: 2.0 }
 	 inner_product_param {
 	   num_output: 8 #按训练集类别改，该值为（类别数+1）*4
 	   weight_filler {
 	     type: "gaussian"
	      std: 0.001
	    }
	    bias_filler {
	      type: "constant"
	      value: 0
	    }
	  }
	}


#### (2)修改py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/stage1_rpn_train.pt和stage2_rpn_train.pt 两个文件

备注:1处修改及其附近的代码


	layer {
	  name: 'input-data'
	  type: 'Python'
	  top: 'data'
	  top: 'im_info'
	  top: 'gt_boxes'
	  python_param {
	    module: 'roi_data_layer.layer'
	    layer: 'RoIDataLayer'
	    param_str: "'num_classes': 2" #按训练集类别改，该值为类别数+1
	  }
	}

#### (3)修改py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt文件

备注:2处修改及其附近的代码

	layer {
	  name: "cls_score"
	  type: "InnerProduct"
	  bottom: "fc7"
	  top: "cls_score"
	  param { lr_mult: 1.0 }
	  param { lr_mult: 2.0 }
	  inner_product_param {
	    num_output: 2 #按训练集类别改，该值为类别数+1
	    weight_filler {
	      type: "gaussian"
	      std: 0.01
	    }
	    bias_filler {
	      type: "constant"
	      value: 0
	    }
	  }
	}
	
	layer {
	  name: "bbox_pred"
	  type: "InnerProduct"
	  bottom: "fc7"
	  top: "bbox_pred"
	  param { lr_mult: 1.0 }
	  param { lr_mult: 2.0 }
	  inner_product_param {
	    num_output: 8 #按训练集类别改，该值为（类别数+1）*4
	    weight_filler {
	      type: "gaussian"
	      std: 0.001
	    }
	    bias_filler {
	      type: "constant"
	      value: 0
	    }
	  }
	}


#### (4)修改py-faster-rcnn/lib/datasets/pascal_voc.py


	self._classes = ('__background__', # always index 0
	                         '你的标签1','你的标签2',你的标签3','你的标签4')

	注:如果只是在原始检测的20种类别:'aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
	'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant',
	'sheep', 'sofa', 'train', 'tvmonitor'中检测单一类别,可参考修改下面的代码:


	def _load_image_set_index(self):
	        """
	        Load the indexes listed in this dataset's image set file.
	        """
	        # Example path to image set file:
	        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
	        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
	                                      self._image_set + '.txt')
	        assert os.path.exists(image_set_file), \
	                'Path does not exist: {}'.format(image_set_file)
	        with open(image_set_file) as f:
	            image_index = [x.strip() for x in f.readlines()]
	
	注:如果需要在原始的20类别只检测车辆的话才需要修改这部分代码.
	        # only load index with cars obj
	        new_image_index = []
	        for index in image_index:
	            filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
	            tree = ET.parse(filename)
	            objs = tree.findall('object')
	            num_objs = 0
	            for ix, obj in enumerate(objs):
	                curr_name = obj.find('name').text.lower().strip()
	                if curr_name == 'car':
	                    num_objs += 1
	                    break
	            if num_objs > 0:
	                new_image_index.append(index)
	        return new_image_index
	
	def _load_pascal_annotation(self, index):
	        """
	        Load image and bounding boxes info from XML file in the PASCAL VOC
	        format.
	        """
	        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
	        tree = ET.parse(filename)
	        objs = tree.findall('object')
	        if not self.config['use_diff']:
	            # Exclude the samples labeled as difficult
	            non_diff_objs = [
	                obj for obj in objs if int(obj.find('difficult').text) == 0]
	            # if len(non_diff_objs) != len(objs):
	            #     print 'Removed {} difficult objects'.format(
	            #         len(objs) - len(non_diff_objs))
	            objs = non_diff_objs
	
	注:如果需要在原始的20类别只检测车辆的话才需要修改这部分代码.
	        # change num objs , only read car
	        # num_objs = len(objs)
	
	        num_objs = 0
	        for ix, obj in enumerate(objs):
	            curr_name = obj.find('name').text.lower().strip()
	            if curr_name == 'car':
	                num_objs += 1
	
	        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
	        gt_classes = np.zeros((num_objs), dtype=np.int32)
	        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
	        # "Seg" area for pascal is just the box area
	        seg_areas = np.zeros((num_objs), dtype=np.float32)
	
	#注:如果需要在原始的20类别只检测车辆的话才需要修改这部分代码
	# Load object bounding boxes into a data 	frame.
	        tmp_ix = 0
	        for ix, obj in enumerate(objs):
	            bbox = obj.find('bndbox')
	            # Make pixel indexes 0-based
	            x1 = float(bbox.find('xmin').text) - 1
	            y1 = float(bbox.find('ymin').text) - 1
	            x2 = float(bbox.find('xmax').text) - 1
	            y2 = float(bbox.find('ymax').text) - 1
	            curr_name = obj.find('name').text.lower().strip()
	            if curr_name != 'car':
	                continue
	            cls = self._class_to_ind[curr_name]
	            boxes[tmp_ix, :] = [x1, y1, x2, y2]
	            gt_classes[tmp_ix] = cls
	            overlaps[tmp_ix, cls] = 1.0
	            seg_areas[tmp_ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
	
	            tmp_ix += 1
	
	        overlaps = scipy.sparse.csr_matrix(overlaps)
	
	        return {'boxes' : boxes,
	                'gt_classes': gt_classes,
	                'gt_overlaps' : overlaps,
	                'flipped' : False,
	                'seg_areas' : seg_areas}

#### (4)py-faster-rcnn/lib/datasets/imdb.py修改


	def append_flipped_images(self):
	        num_images = self.num_images
	        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
	                  for i in xrange(num_images)]
	        for i in xrange(num_images):
	            boxes = self.roidb[i]['boxes'].copy()
	            oldx1 = boxes[:, 0].copy()
	            oldx2 = boxes[:, 2].copy()
	            boxes[:, 0] = widths[i] - oldx2 - 1
	            boxes[:, 2] = widths[i] - oldx1 - 1
	
	            for b in range(len(boxes)):
	                if boxes[b][2] < boxes[b][0]:
	                   boxes[b][0] = 0
	
	            assert (boxes[:, 2] >= boxes[:, 0]).all()


#### (5)py-faster-rcnn/tools/train_faster_rcnn_alt_opt.py修改迭代次数（建议修改）

	max_iters=[8000,4000,8000,4000]
	建议:第一次训练使用较低的迭代次数,先确保能正常训练,如max_iters=[8,4,8,4]

训练分别为4个阶段（rpn第1阶段，fast rcnn第1阶段，rpn第2阶段，fast rcnn第2阶段）的迭代次数。可改成你希望的迭代次数。
如果改了这些数值，最好把py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt里对应的solver文件（有4个）也修改，stepsize小于上面修改的数值，stepsize的意义是经过stepsize次的迭代后降低一次学习率（非必要修改）。

#### (6)删除缓存文件(每次修改配置文件后训练都要做)

	删除py-faster-rcnn文件夹下所有的.pyc文件及data文件夹下的cache文件夹,
	data/VOCdekit2007下的annotations_cache文件夹(最近一次成功训练的
	annotation和当前annotation一样的话这部分可以不删,否则可以正常训练,
	但是最后评价模型会出错)

### (四)开始训练

	cd $FRCN_ROOT
	./experiments/scripts/faster_rcnn_alt_opt.sh 0 ZF pascal_voc


成功训练后在py-faster-rcnn/output/faster_rcnn_alt_opt/voc_2007_trainval文件夹下
会有以final.caffemodel结尾的模型文件,一般为ZF_faster_rcnn_final.caffemodel

成功训练后会有一次模型性能的评估测试,成功的话会有MAP指标和平均MAP指标的输出,类似下文,
训练日志文件保存在experiments/logs文件夹下.

	Evaluating detections
	Writing car VOC results file
	VOC07 metric? Yes
	AP for car = 0.0090
	Mean AP = 0.0090
	~~~~~~~~
	Results:
	0.009
	0.009
	~~~~~~~~
	
	--------------------------------------------------------------
	Results computed with the **unofficial** Python eval code.
	Results should be very close to the official MATLAB eval code.
	Recompute with `./tools/reval.py --matlab ...` for your paper.
	-- Thanks, The Management
	--------------------------------------------------------------
	
	real	1m43.822s
	user	1m25.764s
	sys	0m15.516s

	
### (五)测试训练结果

#### (1)修改py-faster-rcnn\tools\demo.py

	CLASSES = ('__background__',
	         '你的标签1','你的标签2',你的标签3','你的标签4')
	         
	NETS = {'vgg16': ('VGG16',
	                  'VGG16_faster_rcnn_final.caffemodel'),
	        'zf': ('ZF',
	                  'ZF_faster_rcnn_final.caffemodel')}
	                  
	im_names = os.listdir(os.path.join(cfg.DATA_DIR, 'demo'))  


#### (2)放置模型及测试图片

	将训练得到的py-faster-rcnn\output\faster_rcnn_alt_opt\***_trainval中
	ZF的final.caffemodel拷贝至py-faster-rcnn\data\faster_rcnn_models

	测试图片放在py-faster-rcnn\data\demo(与上面demo.py设置路径有关,可修改)


#### (3)进行测试

	cd $FRCN_ROOT
	./tool/demo.py


## 四. 曾出现过的bug及当时的解决方法

(1) 训练时出现KeyError:'max_overlaps'  ,解决方法:删除data文件夹下的cache文件夹

(2) 训练结束后测试时出现类似

	File "/home/hyzhan/py-faster-rcnn/tools/../lib/datasets/voc_eval.py", line 126, in voc_eval
	    R = [obj for obj in recs[imagename] if obj['name'] == classname]
	KeyError: '000002'

解决方法: 删除data/VOCdekit2007下的annotations_cache文件夹

(3) caffe-fast-rcnn编译时出现找不到nvcc命令的情况,解决方法:

	export PATH=/usr/local/cuda-8.0/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

将cuda安装路径添加到环境变量中

(4) caffe-fast-rcnn编译时出现类似找不到opencv命令的情况,解决方法,添加环境变量:

	export LD_LIBRARY_PATH=/home/hyzhan/software/opencv3/lib:$LD_LIBRARY_PATH

(5) 训练的时候执行"./experiments/scripts/faster_rcnn_alt_opt.sh 0 ZF pascal_voc"语句进行训练会出现找不到faster_rcnn_alt_opt.sh文件的情况,解决方法:重新手打命令

(6) 测试之前需要修改tool文件夹下的demo或者mydemo里面的class类别,不然会显示上次训练的类别
