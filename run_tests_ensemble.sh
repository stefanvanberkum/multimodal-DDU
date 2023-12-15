#!/bin/bash

# Run all tests for the trained ensembles.

for train_ds in cifar10 cifar100
do
  for mode in accuracy ece ood
  do
    if [[ $mode == ood ]]
    then
      for test_ds in cifar100 svhn imageNet
      do
        if ! [[ $train_ds == cifar100 ]] || ! [[ $test_ds == cifar100 ]]
        then
          for uncertainty in entropy mi
          do
            echo " "
            echo "--train_ds $train_ds --test_ds $test_ds --test $mode --uncertainty $uncertainty"
            echo "--------"
            python run_tests_ensemble.py --model wrn-ensemble --train_ds $train_ds --test_ds $test_ds --test $mode --uncertainty $uncertainty
            wait
            echo "--------"
            echo " "
          done
        fi
      done
    else
      echo " "
      echo "--train_ds $train_ds --test_ds $train_ds --test $mode"
      echo "--------"
      python run_tests_ensemble.py --model wrn-ensemble --train_ds $train_ds --test_ds $train_ds --test $mode
      wait
      echo "--------"
      echo " "
    fi
  done
done
