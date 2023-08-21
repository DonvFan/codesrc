#pragma once
#include<vector>
#include<iostream>
#include<random>
#include<chrono>
#include<string>
#include<algorithm>
#include<thread>
#include "ptable.h"
using namespace std;

class BigInt
{
    private:
    using base_t = unsigned long long;
    const static unsigned int base_bit_num = 64; 
    const static unsigned int base_bit_expo = 6;
    const static unsigned int base_bit_remi = 0x3F;
    const static base_t base = ULLONG_MAX;
    // using base_t = unsigned long ;
    // const static unsigned int base_bit_num = 32; 
    // const static unsigned int base_bit_expo = 5;
    // const static unsigned int base_bit_remi = 0x1f;
    // const static base_t base = ULONG_MAX;
    // const static BigInt pt[];

    bool is_positive;
    vector<base_t> data;
    unsigned int data_bit_num;

    BigInt& abs_add(const BigInt& b);
    BigInt& abs_sub(const BigInt& b);
    BigInt& abs_multi(const BigInt& b);
    BigInt& abs_mod(const BigInt& b,  BigInt& quot);
    BigInt& leftShiftInplace(const unsigned int b);
    BigInt& rightShiftInplace(const unsigned int b);
    bool abs_large(const BigInt& b) const;
    bool abs_largeOrEqual(const BigInt& b) const;
    void count_bit_nums();
    void trim();

    char hexTochar(char ch)
	{
		static char table[]={0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
		if(isdigit(ch))
			ch-='0';
		else if(islower(ch))
			ch-='a'-10;
		else if(isupper(ch))
			ch-='A'-10;

		return table[ch];	
	}


    public:
    friend BigInt operator + (const BigInt& a, const BigInt& b);
    friend BigInt operator - (const BigInt& a, const BigInt& b);
    friend BigInt operator * (const BigInt& a, const BigInt& b);
    friend BigInt operator / (const BigInt& a, const BigInt& b);
    friend BigInt operator % (const BigInt& a, const BigInt& b);
    friend BigInt operator << (const BigInt& a, const unsigned int bit_num);
    friend BigInt operator >> (const BigInt& a, const unsigned int bit_num);
    friend bool operator == (const BigInt& a, const BigInt& b);
    friend bool operator != (const BigInt& a, const BigInt& b);
    friend bool operator > (const BigInt& a, const BigInt& b);
    friend bool operator >= (const BigInt& a, const BigInt& b);
    friend bool operator <= (const BigInt& a, const BigInt& b);
    friend bool operator <(const BigInt& a, const BigInt& b);
    // void operator =(const BigInt& a)
    // {
    //     data = a.data;
    //     data_bit_num = b
    // }

    const static BigInt ZERO;
    const static BigInt ONE;
    const static BigInt TWO;

    BigInt(const BigInt& a):
        is_positive(a.is_positive), data(a.data), data_bit_num(a.data_bit_num){}

    BigInt():is_positive(true), data_bit_num(0), data(0){}

    BigInt(base_t value, bool is_posi = true): data(1, value), is_positive(is_posi)
    {
        count_bit_nums();
    }

    BigInt(string hexstr, bool is_litte_endian = false)
    {
        if(hexstr.front() == '-')
        {
            this->is_positive = false;
            hexstr = hexstr.substr(1);
        }    
        else 
            this->is_positive = true;
        
        unsigned int shift = 4;
        string temp;
        int hex_bit_num = base_bit_num / shift;
        unsigned int addZero =hex_bit_num -  hexstr.length() % hex_bit_num;
        while (addZero > 0)
        {
            temp.push_back('0');
            addZero--;
        }
        temp += hexstr;
        char c = 0;
        base_t ele = 0;
        for(int i = 0; i<temp.size(); )
        {
            for(int j = 0; j < hex_bit_num; j++, i++)
            {
                c = hexTochar(temp[i]);
                ele = ((ele << shift) | c);
            }

            data.push_back(ele);
            ele = 0;
        }

        reverse(data.begin(), data.end());
        trim();
        
    }

    ~BigInt()
    {

    }

    BigInt modexp(const BigInt& base, const BigInt& power) const;
    BigInt exGCD(const BigInt& m);
    string toHexStr()
    {
        static char hex[]={'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
        string res ="";

        base_t bits_flag = 0x0F;
        for(int i = 0; i<data.size(); i++)
        {
            base_t temp =  data[i];
            for(int j = 0; j<base_bit_num; j+=4)
            {
                res.push_back(hex[bits_flag & temp]);
                temp >>= 4;
            }

        }

        for(int i = res.size() - 1; i>0; i--)
        {
            if(res.back() == '0')
                res.pop_back();
        }
        if(!is_positive)
            res.push_back('-');
        reverse(res.begin(), res.end());
        return res;
    }

    static BigInt bigPrimeGen(const unsigned int n, const unsigned int MRtest_round = 7, const unsigned int small_round =100)
    {
        BigInt res;            
        bigOddGen(n, res);
        while(!( SPTest(res, small_round)&&MRtestParallel(res, MRtest_round)))
        {
            // cout<<res.toHexStr()<<endl;
            // res.abs_add(TWO);
            res.data.front() += 2;
        }
//        cout<<res.toHexStr()<<endl;
//        cout<<"OK"<<endl;
        return res;
    }

    //Todo: use a reference parameter as output
    static void bigOddGen(const unsigned int n, BigInt& output)
    {
        static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        static default_random_engine dre(seed);
        static uniform_int_distribution<base_t> mid_uid(0, base);
        unsigned int tail_big_num = n % 64;
        base_t one = 1;
        //BigInt res; //Maybe we can set vector preserve len;
        base_t upper = one << n, lower = one <<(n - 1);
        upper--;
        uniform_int_distribution<base_t> tail_uid(lower, upper);
        base_t  temp;

        for(int i = base_bit_num; i<n; i += base_bit_num)
        {
            temp = mid_uid(dre);
            if(i == base_bit_num && temp % 2 == 0)
                temp += 1;
            output.data.push_back(temp);
        }
        output.data.push_back(tail_uid(dre));
        if(output.data.size() == 1 && output.data.back() % 2 == 0)
            output.data.back() += 1;

        output.data_bit_num = n;
    }

    
    // too small
    static BigInt smallerRandGen(const BigInt& num)
    {
        static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        static default_random_engine dre(seed);
        static uniform_int_distribution<base_t> mid_uid(2, base);  
        base_t value = 0;
        value = mid_uid(dre);
        BigInt res(value);
        return res;
    }

    static bool SPTest(const BigInt& n, unsigned int iter_num)
    {
        for(int i = 0; i<iter_num; i++)
        {
            if(n % BigInt(ptable[i]) == ZERO)
                return false;
        }
        return true;
    }

    static bool MRtest(const BigInt& n, unsigned int iter_round)
    {
        if(n == TWO)
            return true;

        BigInt n_sub_1 = n - ONE;
        // if(n_sub_1.at(0))
        //     return false;

        while(iter_round > 0)
        {
            iter_round--;
            BigInt a = smallerRandGen(n_sub_1);
            BigInt d = ONE;
            BigInt x = d;
            //No r and d special？ 2^r*d + 1 = n?
            for(int i = n_sub_1.data_bit_num - 1; i>=0; i--)
            {
                x = d;
                d = (d * d) % n;
                if(d == BigInt::ONE && x != BigInt::ONE && x != n_sub_1)
                    return false;

                if(n_sub_1.at(i))
                    d = (a * d) % n;
            }
            if(d != ONE)
                return false;
        }

        return true;
    }

    static bool MRtestParallel(const BigInt& n, const unsigned int iter_round)
    {
        if(n == TWO)
            return true;
        BigInt n_sub_1 = n - ONE;
        bool flags[iter_round];
        thread threads[iter_round];

        for(int i = 0; i<iter_round; i++)
        {
            cout<<i<<endl;
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

    static bool MRtest2(const BigInt& n, unsigned int iter_round = 7)
    {
        if(n == TWO)
            return true;
        BigInt n_sub_1 = n - ONE;
        unsigned int r = 0;
        for(int i = 0; i<n_sub_1.data_bit_num; i++)
        {
            if(n_sub_1.at(i) == 0)
                r++;
        }
        BigInt d = n_sub_1 >> r;
        BigInt a;
        while(iter_round-- > 0)
        {
            a = smallerRandGen(n_sub_1);
            BigInt x = ONE;
            for(int i = d.data_bit_num - 1; i>=0; i--)
            {
                x = (x * x) % n;
                if(d.at(i) > 0)
                    x = (a * x) % n;
            }
            if(x == ONE || x == n_sub_1)
                continue;

            for(int i = 1; i < r; i++)
            {
                x = (x * x) % n;
                if (x == n_sub_1)
                    continue;
            }
            return false;
        }
        return true;
    }
    

    bool at(std::size_t n) const
    {
        if (n >= data_bit_num)
            return false;

        std::size_t index = n >> (base_bit_expo);
        std::size_t offset = n & (base_bit_remi);
        base_t t = data[index];
        base_t one = 1;
        return (t & (one<<offset));
    }


};
