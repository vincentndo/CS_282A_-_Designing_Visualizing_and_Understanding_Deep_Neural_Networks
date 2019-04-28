#!/bin/bash

python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 1000 -lr 0.002 --exp_name pend_b_1000_lr_0.002;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 2000 -lr 0.002 --exp_name pend_b_2000_lr_0.002;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 3000 -lr 0.002 --exp_name pend_b_3000_lr_0.002;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 4000 -lr 0.002 --exp_name pend_b_4000_lr_0.002;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 5000 -lr 0.002 --exp_name pend_b_5000_lr_0.002;

python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.0005 --exp_name pend_b_6000_lr_0.0005;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.0008 --exp_name pend_b_6000_lr_0.0008;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.001 --exp_name pend_b_6000_lr_0.001;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.002 --exp_name pend_b_6000_lr_0.002;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.003 --exp_name pend_b_6000_lr_0.003;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.004 --exp_name pend_b_6000_lr_0.004;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.005 --exp_name pend_b_6000_lr_0.005;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.007 --exp_name pend_b_6000_lr_0.007;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 6000 -lr 0.01 --exp_name pend_b_6000_lr_0.01;

python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 7000 -lr 0.001 --exp_name pend_b_7000_lr_0.001;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 7000 -lr 0.002 --exp_name pend_b_7000_lr_0.002;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 7000 -lr 0.003 --exp_name pend_b_7000_lr_0.003;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 7000 -lr 0.004 --exp_name pend_b_7000_lr_0.004;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 7000 -lr 0.005 --exp_name pend_b_7000_lr_0.005;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 7000 -lr 0.007 --exp_name pend_b_7000_lr_0.007;

python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 8000 -lr 0.001 --exp_name pend_b_8000_lr_0.001;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 8000 -lr 0.002 --exp_name pend_b_8000_lr_0.002;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 8000 -lr 0.003 --exp_name pend_b_8000_lr_0.003;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 8000 -lr 0.004 --exp_name pend_b_8000_lr_0.004;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 8000 -lr 0.005 --exp_name pend_b_8000_lr_0.005;
python train_pg.py Pendulum-v0 -ep 1000 --discount 0.99 -n 400 -e 3 -rtg -b 8000 -lr 0.007 --exp_name pend_b_8000_lr_0.007;
