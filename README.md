# Simple-Neutral-Network-on-Dataset-CIFAR10
A simple nn model dividing some pictures into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
The average accuracy of the model "model_40.pth" is around 66% (Having trained for 40 epoches)

[READ] Attention:
1. The instruction of constructing the environment of this programme: https://www.bilibili.com/list/watchlater?oid=74281036&bvid=BV1hE411t7RN

2. If you wanna to use your GPU-RTX50+ to train the model (having applied the code "XXX.coda()"), ensure that the version of the Coda is no less than 12.8.
   If the version does not satisfy, please download Conda 12.8 from the website:

3. After tip 2, Ensure that the version of Pytorch meets the requirement of Conda.
   If not, input these two commands into your terminal to update your Pytorch to 12.8:
       (1) pip uninstall torch torchvision torchaudio
       (2) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

4. If you meet any obstacles or bugs in my programme, please report them to me, thanks!
