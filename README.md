## Edge Eval Python
A python implementation of [edge eval](https://github.com/s9xie/hed_release-deprecated/tree/master/examples/eval).

The logic of the code is almost the same as that of the origin MATLAB implementation (see [References](#References)).
The core code is used directly  [edge-eval-python](https://github.com/Walstruzz/edge_eval_python).

I just added some optimizations for ease of use. Currently, BSDS, NYUD, BIPED, UDED can be used in one click, and other datasets can be easily eval by specifying GT.

## Requirements
* Python3
* Numpy
* Scipy >= 1.6.0
* Opencv
* tqdm
* Matplotlib
* g++

## Operating System Compatibility
This project supports both Linux and Windows environments. 

But for Windows users, you must install WSL (Windows Subsystem for Linux). 

#### Step-by-Step WSL Installation:
### 1. Check virtualization support
* Open Task Manager → Performance → CPU, and confirm that Virtualization is enabled.
* If it is disabled, you may need to enable it in your BIOS/UEFI settings. For guidance, follow this official article: [Enable virtualization on Windows](https://support.microsoft.com/es-es/windows/habilitar-la-virtualizaci%C3%B3n-en-windows-c5578302-6e43-4b4b-a449-8ced115f58e1)

### 2. Enable Windows features
* Open Control Panel → Programs and Features → Turn Windows features on or off.
* Enable the option: Windows Subsystem for Linux
* Restart your computer if prompted.

### 3. Enable WSL (one-time setup)
Open PowerShell as Administrator and run:
``` shell
wsl --install
```
If you already have WSL installed, make sure it’s WSL2. You can check with:
``` shell
wsl --list --verbose
```
If WSL2 in not installed. You can update it with:
``` shell
wsl --set-version Ubuntu 2
```

### 3. Install Ubuntu
By default, "wsl --install" installs the latest Ubuntu version. You can also manually install other Ubuntu version from the Microsoft Store.

### 4. Launch Ubuntu and finish setup:
Run "wsl" in your terminal or open your Ubuntu version from the Start Menu. Set your username and password when prompted.

### 5. Access the root directory of your WSL distribution:
Open your Ubuntu terminal and run the following commands:
``` shell
cd
explorer.exe .
```
This will open the current WSL directory in Windows Explorer, making it easier to copy project files into WSL.

### 6. Update your packages
Inside the Ubuntu terminal:
``` shell
sudo apt update && sudo apt upgrade
```

## Generic Install
### 1. Clone repository
``` shell
git clone https://github.com/xavysp/eval-edge-pytorch_xsp.git
cd eval-edge-pytorch_xsp-main
```
After this we recommend to create and use a python virtual environment

### 2. Compile cxx library
Most of the code in this folder is copied from [davidstutz/extended-berkeley-segmentation-benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark/tree/master/source).

Actually, there is a more efficient function in `Scipy` that can solve the CSA problem without compiling the following cxx codes...
``` shell
cd cxx/src
source build.sh
```

## Usage
### 1. Save your results 
Save your result like this: [https://github.com/Li-yachuan/CTFN-pytorch-master/blob/main/test.py](https://github.com/Li-yachuan/CTFN-pytorch-master/blob/main/test.py).


Create a `/result/"model-dataset"` folder. And make sure to create two folders `mat` and `png` in that folder to save the two forms of the prediction results.
Otherwise, this app will create for you and tel you that the mat or png folder is needed. 

Then the folder structure should be:  
``` 

\result\TEED-UDED  
....|-png  
....|...|-a.png  
....|...|-b.png  
....|....:  
....|-mat  
........|-a.mat    
........|-b.mat  
		:
```

### 2. Download GT
Create a `\GT\dataset"\gt` folder, for example: `\GT\UDED\gt`

Download a project and unzip the GT in the folder `\GT\"dataset"`

They are available [here](https://drive.google.com/drive/folders/1j1TU28PinKipOh0egf8tbzI7EetAbzKh?usp=sharing)


### 3.Evaluation

#### 3.1  The simplest eval
We have a little change here, remember, before running the evaluation 
you should prepare your dataset as" `GT/"DATASET"/"gt_dir""`, the DATASET and
gt_dir is for your project.
 Once the GT dirs are finished setting, we start with 
the results folder, `result/"Model-DATASET"/` then `[png, nms]` folders.
We are going work with TEED model  and UDED dataset, so I have prepared:
`GT/UDED/gt_ed` and `result/TEED-UDED/[png,nms]`, see the shell:
``` shell
python eval.py -d UDED -m TEED  -gt gt_ed -nms -f
```
`-nms` TEED needs to apply NMS before evaluation and `-f` with the full
evaluation performance. `-d` is your dataset, `-m` your model.
`-gt` is the gt dir that is into your GT then dataset folder, by default it should be 
"gt". We provide another example with [MarchED](https://github.com/Bedrettin-Cetinkaya/MatchED) model,
as this edge detector does not need NMS so it should run as:

``` shell
ython eval.py -d UDED -m MatchED  -gt gt_ed  -f
```

At the end, your dir `result/MatchED-UDED` will have the following dirs: `nms, png, nms-eval`. 

For higher automation, we implement automatic detection of folders with simple cycle. We use the `T` and `limit` parameters to control the sleep time and total wait time for the folder traversal, the default of which is 0.5 hours and 8 hours.

#### 3.2 No constant monitoring
If you only need to eval existing results, you can use `notwait` to cancel the automatic check and keep waiting, as follows:


``` shell
python eval.py \result -d BSDS -nw -f
```

#### 3.3 Default dataset

If you can include the dataset name in the path under eval, then you don't have to specify the dataset individually:


``` shell
python eval.py \result\bsds-result -nw -f
```

#### 3.4 Multiple paths

You can specify multiple paths to eval multiple sets of results at the same time.You need to place multiple paths inside the "", separated by Spaces. There is no support for evaling multiple datasets at the same time, so multiple results should be on the same dataset.		

``` shell
python eval.py "\result\bsds-result \result\bsds-result2" -nw -f
```

#### 3.5 Light version & full version

Evaluation of edge detection is known to be a very time-consuming process, in extreme cases even slower than training, such as on densely labeled datasets like BIPED.


To speed up eval, we divided eval into two versions, light and full, by controlling the sampling frequency of the threshold. The full version is currently the most commonly used thrs=99, and thrs=9 in light.

light version:
``` shell
python eval.py "\result\bsds-result \result\bsds-result2" -nw
```

full version:
``` shell
python eval.py "\result\bsds-result \result\bsds-result2" -nw -f
```

The light version is faster, but the accuracy will be about 0.5% lower than the full version, but the relative accuracy will not change, so we can use the light version to pick the best result, and then use the full version to get its true accuracy.

#### 3.6 Show result

To see the results better, use the following command.

For full version:		

``` shell
python show.py "\result\bsds-result" -f
```

For light version:		

``` shell
python show.py "\result\bsds-result"
```

## Note
* The edges of the image are 1 (white) and the background is 0 (black).


## Note (same as  [edge-eval-python](https://github.com/Walstruzz/edge_eval_python).)
* Because of the difference in calculation precision and the sensitivity of NMS threshold, the edge images may be **slightly** different.
* `match_edge_maps` samples points randomly (**so as Matlab**).
* Python and Matlab index files in different order, resulting in different order of `eval_bdry_img.txt`.
* Python version is slower than Matlab version. Should I implement more functions in `cxx/lib/solve_cas.so`?

## References
* [edge eval](https://github.com/s9xie/hed_release-deprecated/tree/master/examples/eval)
* [extended-berkeley-segmentation-benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark)
* [bwmorph_thin](https://gist.github.com/joefutrelle/562f25bbcf20691217b8)
* [pdollar's image & video Matlab toolbox ](https://github.com/pdollar/toolbox)
* [pdollar's edge detection toolbox](https://github.com/pdollar/edges)
* [PyTorch Reimplementation of HED](https://github.com/xwjabc/hed)
* [edge-eval-python](https://github.com/Walstruzz/edge_eval_python)

