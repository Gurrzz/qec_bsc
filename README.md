# qec_bsc
Code repository connected to our BSc Quantum Error Correction project


This is an extention of the qecsim package, with large inspiration from 
the qsdxzzx extention package to qecsim, created with the ambition to 
implement a decoder for the XZXZ/ZXZX code (shortend to G81), an example 
of which can be seen below: 


                 -------
                /       \
               |Z (0,2) X|
               +---------+---------+-----
               |X       Z|X       Z|X    \
               |  (0,1)  |  (1,1)  |(2,1) |
               |X       Z|X       Z|X    /
          -----+---------+---------+-----
         /    X|Z       X|Z       X|
        |(-1,0)|  (0,0)  |  (1,0)  |
         \    X|Z       X|Z       X|
          -----+---------+---------+
                         |X       Z|
                          \ (1,-1)/
                           -------



This code is created by starting out with the XZ code and applying 
Hadamard operators (H) in vertical lines in every second column. 


#Links 

[qecsim](https://github.com/qecsim)
[qsdxzzx](https://bitbucket.org/qecsim/qsdxzzx/src/master/)

