#pragma once
#include"BigInt.h"

class RSA
{
    private:
        BigInt p, q, n;
        BigInt e, d, phi_n;

        void static wrapGenBigPrime(const unsigned int bit_num, BigInt* res);

    public:
        RSA(const unsigned int bit_num);
        ~RSA();
        pair<BigInt, BigInt> getPubKey();
        BigInt encrypt(const BigInt& M);
        BigInt decrypt(const BigInt& C);
//        string get_p()
//        {
//            return p.toHexStr();
//        }
//        string get_q()
//        {
//            return q.toHexStr();
//        }
        string get_e()
        {
            return e.toHexStr();
        }
        string get_d()
        {
            return d.toHexStr();
        }

        string get_n()
        {
            return n.toHexStr();
        }
};
//extern "C"
//{
//    RSA* get_rsa(unsigned int num)
//    {
//        return new RSA(num);
//    }
//}
