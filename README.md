## blackout

torch  1.4.0 
python 3.5


python main_cot.py --gpu 4 --sess 100black2_50_0.01w --cifar 100 -e 200 --lr 0.001 --k 50 --black

--gpu    wich gpu to use


--sess   name to be saved as


--cifar  (0,10,100) specify which dataset( mnist, cifar10,cifar100)

--black enable blackout

--GCE enbale COT training

if use code black1 ( target dependent sampling), lerning rate should be 0.01
