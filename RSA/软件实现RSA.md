# 软件实现RSA

## 1 程序使用说明

本实验使用C++实现了RSA对随机输入字符串的加解密，并设计实现了简洁的图形化界面提高了工具的易用性。运行步骤如下：

1) 点击运行 ***.\bin\RSA-GUI.exe***，会弹出如下图形界面：
   
   ![image-20211108102238761](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20211108102238761.png)

2) 首先在下拉框选择模N长度，目前可以选择RSA-128，RSA256，RSA-512，RSA-768和RSA-1024五种长度：
   
   ![image-20211108102342933](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20211108102342933.png)
3. 选择完成后，点击``RSA初始化``生成公钥私钥，生成成功后会弹出窗口提醒：
   
   ![image-20211108102517536](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20211108102517536.png)

4. **加密：**
   
   * 在`输入文本框`中随机输入字符串，点击`加密`，在`输出文本框`中可得到相应的加密输出。
   
   ![image-20211108104344635](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20211108104344635.png)

5. **解密：**
   
   * 在``输入文本框``中输入加密后的字符串，点击``解密``，在``输出文本框``即可得到相应加密输出。
     
     ![image-20211108104302233](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20211108104302233.png)

6. 运行效率
   
   * **测试环境：** Window10 专业版，Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz   2.90 GHz。
   
   * **测试方法**：单独运行10次RSA生成密钥程序，运行时间取平均值。
   
   * **测试结果：**如下表：
   
   | 模N长度 | RSA-128 | RSA-256 | RSA-512 | RSA-768 | RSA-1024 |
   | ---- | ------- | ------- | ------- | ------- | -------- |
   | 运行时间 | 0.1s    | 0.3s    | 0.7s    | 1.0s    | 2.3s     |

## 2 算法/代码

* C++自主实现大数类，使用unsigned long long存储数组存储二进制数据，使用移位运算实现大数乘除法。

* 多线程并行实现p、q大数和Miller Rabin检测。
  
  ```C++
      static bool MRtestParallel(const BigInt& n, const unsigned int iter_round)
      {
          if(n == TWO)
              return true;
          BigInt n_sub_1 = n - ONE;
          bool flags[iter_round];
          thread threads[iter_round];
  
          for(int i = 0; i<iter_round; i++)
          {
              threads[i] = thread(MRTestOnce, &n, &n_sub_1, &flags[i]);
          }
  
          for(int i = 0; i<iter_round; i++)
          {
              threads[i].join();
          }
  
          for(int i = 0; i<iter_round; i++)
          {
              if(! flags[i])
                  return false;
          }
  
          return true;
  
      }
  
      static void MRTestOnce(const BigInt* n, const BigInt* n_sub_1, bool* flag)
      {
          BigInt a = smallerRandGen(*n_sub_1);
          BigInt d = ONE;
          BigInt x = d;
          //No r and d special？ 2^r*d + 1 = n?
          for(int i = n_sub_1->data_bit_num - 1; i>=0; i--)
          {
              x = d;
              d = (d * d) % *n;
              if(d == BigInt::ONE && x != BigInt::ONE && x != *n_sub_1)
               {
                  *flag = false;
                  return;
              }
  
              if(n_sub_1->at(i))
                  d = (a * d) % *n;
          }
          if(d != ONE)
          {
              *flag = false;
              return;
          }
          *flag = true;
      };
  ```

* Miller Rabin检测前先使用长度为n的素数表对随机生成大数是否为素数进行检查，对于不同的模N长度使用不同n进行检测并测试速度。其中RSA-128、RSA-256使用n = 10的素数表时运行速度最快，其他长度下，RSA-512对应n=50， RSA-768与RSA-1024时对应n=100。

* Miller Rabin算法改动：
  
  原算法伪代码如下：
  
  ```c
  Input #1: n > 3, an odd integer to be tested for primality
  Input #2: k, the number of rounds of testing to perform
  Output: “composite” if n is found to be composite, “probably prime” otherwise
  
  write n as 2^r·d + 1 with d odd (by factoring out powers of 2 from n − 1)
  WitnessLoop: repeat k times:
      pick a random integer a in the range [2, n − 2]
      x ← a^d mod n
      if x = 1 or x = n − 1 then
          continue WitnessLoop
      repeat r − 1 times:
          x ← x^2 mod n
          if x = n − 1 then
              continue WitnessLoop
      return “composite”
  return “probably prime”
  ```
  
  直接计算 $ a^d\mod n$因为d较大会花费较多时间，此处是基于费马小定理$a^{n-1}\equiv1(\mod n)$，在计算$a^{n-1}(\mod n)$中使用了如下算法，只需要循环r+1次，远小于d：
  
  ```c++
  n = b1*2^r + b2*2^(r-1) + ....+b_k*2^0;
  d = 1;
  x = 1;
  for i in range(1, r+1):
      x = d;
      d = (d * d)%n
      if b_i:
          d = (a * d) %n
  ```
