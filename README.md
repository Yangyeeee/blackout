## blackout

torch  1.4.0 
python 3.5

baseline:

python main_cot.py --gpu 4 --sess softmax.1 --cifar 100 -e 200 --lr 0.1    (test acc: 74.11%)


blackout:

python main_cot.py --gpu 4 --sess black1.50.1 --cifar 100 -e 200 --lr 0.1 --k 50 --blackout 1    (test acc: 75.28%)

python main_cot.py --gpu 4 --sess black1.25.1 --cifar 100 -e 200 --lr 0.1 --k 25 --blackout 1    (test acc: 75.32%)

python main_cot.py --gpu 4 --sess black3.25.1 --cifar 100 -e 200 --lr 0.1 --k 25 --blackout 3   (nonuniform sampling)

--gpu    wich gpu to use


--sess   name to be saved as


--cifar  (0,10,100) specify which dataset( mnist, cifar10,cifar100)

--blackout (0 , 1, 2, 3) enable different blackout versions (1 best so far, 3 is nonuniform sampling)

--GCE enbale COT training

