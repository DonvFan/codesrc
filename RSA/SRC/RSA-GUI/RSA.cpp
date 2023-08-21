#include "RSA.h"
#include "BigInt.h"
#include<random>
#include<time.h>
#include<thread>
#include<future>
using namespace std;

void wrapMulti(const BigInt& p, const BigInt& q, const & res)
{

}
RSA::RSA(const unsigned int bit_num)
{
    static default_random_engine dre(time(NULL));
    static uniform_int_distribution<unsigned int> uid(0, 5000);
//    BigInt& pt = p;
//    BigInt& qt = q;
    thread t1(wrapGenBigPrime, bit_num, &p);
    thread t2(wrapGenBigPrime, bit_num, &q);
    t1.join();
    t2.join();
    cout<<"done"<<endl;
    n = p * q;
    phi_n = (p - BigInt::ONE) * (q - BigInt::ONE);
    e = ptable[uid(dre)];
    d = e.exGCD(phi_n);
}

RSA::~RSA()
{

}

BigInt RSA::encrypt(const BigInt& m)
{
    return m.modexp(e, n);
}


BigInt RSA::decrypt(const BigInt& c)
{
    return c.modexp(d, n);
}


void  RSA::wrapGenBigPrime(const int unsigned bit_num, BigInt* res)
{
    *res = BigInt::bigPrimeGen(bit_num);
}

