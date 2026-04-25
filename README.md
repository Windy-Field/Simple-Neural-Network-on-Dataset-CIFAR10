# Simple-Neutral-Network-on-Dataset-CIFAR10
A very simple NN model dividing Dataset CIFAR10 into 10 different categories [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck].
Studying from the course: https://www.bilibili.com/list/watchlater?oid=74281036&bvid=BV1hE411t7RN.
The average accuracy of the model "model_40.pth" is around 66% (Having trained for 40 epoches).

[README] Attention:
1. The instruction of constructing the environment of this programme: https://www.bilibili.com/list/watchlater?oid=74281036&bvid=BV1hE411t7RN.

2. After constructing the environment, you can download these 3 files: "Train.py", "Test.py" and "model_40.pth" to run and play.
   Note that you shall also create a folder named "imgs" in your running folder and prepare some test pictures (dogs, cats, aeroplanes etc.) beforehand.
   The other "XXX.py" files are my study notes, which are not necessary.

3. If you wanna to use your GPU-RTX50+ to train the model (having applied the code "XXX.coda()"), ensure that the version of the Cuda is no less than 12.8.
   If the version does not satisfy, please download Cuda 12.8 from the website: https://developer.nvidia.com/cuda-12-8-0-download-archive

4. After Step 3, Ensure that the version of Pytorch meets the requirement of Cuda.
   If not, input these 2 commands into your terminal to update your Pytorch to 12.8:
       (1) pip uninstall torch torchvision torchaudio
       (2) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

5. If you meet any other obstacles or bugs in my programme, please report them to me, thanks!
