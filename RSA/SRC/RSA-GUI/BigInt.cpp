#pragma once
#include "BigInt.h"
#include <thread>
using namespace std;

const BigInt BigInt::ZERO(0);
const BigInt BigInt::ONE(1);
const BigInt BigInt::TWO(2);

// const static BigInt::BigInt pt []= {
//     BigInt(3),BigInt(5),BigInt(7),BigInt(11),BigInt(13),BigInt(17),BigInt(19),BigInt(23),BigInt(29
// ),
// BigInt(31),BigInt(37),BigInt(41),BigInt(43),BigInt(47),BigInt(53),BigInt(59),BigInt(61),BigInt(67),BigInt(71
// ),
// BigInt(73),BigInt(79),BigInt(83),BigInt(89),BigInt(97),BigInt(101),BigInt(103),BigInt(107),BigInt(109),BigInt(113
// ),
// BigInt(127),BigInt(131),BigInt(137),BigInt(139),BigInt(149),BigInt(151),BigInt(157),BigInt(163),BigInt(167),BigInt(173
// ),
// BigInt(179),BigInt(181),BigInt(191),BigInt(193),BigInt(197),BigInt(199),BigInt(211),BigInt(223),BigInt(227),BigInt(229
// ),
// BigInt(233),BigInt(239),BigInt(241),BigInt(251),BigInt(257),BigInt(263),BigInt(269),BigInt(271),BigInt(277),BigInt(281
// ),
// BigInt(283),BigInt(293),BigInt(307),BigInt(311),BigInt(313),BigInt(317),BigInt(331),BigInt(337),BigInt(347),BigInt(349
// ),
// BigInt(353),BigInt(359),BigInt(367),BigInt(373),BigInt(379),BigInt(383),BigInt(389),BigInt(397),BigInt(401),BigInt(409
// ),
// BigInt(419),BigInt(421),BigInt(431),BigInt(433),BigInt(439),BigInt(443),BigInt(449),BigInt(457),BigInt(461),BigInt(463
// ),
// BigInt(467),BigInt(479),BigInt(487),BigInt(491),BigInt(499),BigInt(503),BigInt(509),BigInt(521),BigInt(523),BigInt(541
// ),
// BigInt(547),BigInt(557),BigInt(563),BigInt(569),BigInt(571),BigInt(577),BigInt(587),BigInt(593),BigInt(599),BigInt(601
// ),
// BigInt(607),BigInt(613),BigInt(617),BigInt(619),BigInt(631),BigInt(641),BigInt(643),BigInt(647),BigInt(653),BigInt(659
// ),
// BigInt(661),BigInt(673),BigInt(677),BigInt(683),BigInt(691),BigInt(701),BigInt(709),BigInt(719),BigInt(727),BigInt(733)
// };


void BigInt::trim()
{
    int s = data.size();
    // base_t zero = 0;
    for(int i = s - 1; i>0; i--)
    {
        if(data[i] == 0)
            data.pop_back();
        else
            break;
    }
    count_bit_nums();
}

BigInt BigInt::exGCD(const BigInt& m)
{
    BigInt a[3],b[3],t[3];
	a[0] = ONE; a[1] = ZERO; a[2] = m;
	b[0] = ZERO; b[1] = ONE; b[2] = *this;
	if (b[2] == ZERO || b[2]== ONE)
		return b[2];

	while(true) 
	{
		if (b[2] == ONE) 
		{
			if(b[1].is_positive == false)
				b[1]=(b[1]%m+m)%m;
			return b[1];
		}

		BigInt q = a[2]/b[2];
		for(int i=0; i<3;++i)
		{
			t[i] = a[i] - q * b[i];
			a[i] = b[i];
			b[i] = t[i];
		}
	}
}

bool BigInt::abs_large(const BigInt& b) const
{
    if(b.data_bit_num == data_bit_num)
    {
        for(int i = data.size() - 1; i>0; i--)
        {
            if(data[i] != b.data[i])
                return data[i] > b.data[i];
        }
        return data[0] > b.data[0];
    }

    return data_bit_num > b.data_bit_num;
}

bool BigInt::abs_largeOrEqual(const BigInt& b) const
{
    if(b.data_bit_num == data_bit_num)
    {
        for(int i = data.size() - 1; i>0; i--)
        {
            if(data[i] != b.data[i])
                return data[i] > b.data[i];
        }
        return data[0] >= b.data[0];
    }
    return data_bit_num > b.data_bit_num;
}


bool operator == (const BigInt& a, const BigInt& b)
{
    if(a.is_positive == b.is_positive && 
        a.data_bit_num == b.data_bit_num)
    {
        for(int i = a.data.size() - 1; i>=0; i--)
        {
            if(a.data[i] != b.data[i])
                return false;
        }

        return true;
    }
    return false;
}


bool operator != (const BigInt& a, const BigInt& b)
{
    return !(a == b);
}


bool operator > (const BigInt& a, const BigInt& b)
{
    if(a.is_positive != b.is_positive)
    {
        return a.is_positive > b.is_positive;
    }
    
    if(a.is_positive)
        return a.abs_large(b);
    return !(a.abs_largeOrEqual(b));

}


bool operator >= (const BigInt& a, const BigInt& b)
{
    if(a.is_positive != b.is_positive)
    {
        return a.is_positive > b.is_positive;
    }
    
    if(a.is_positive)
        return a.abs_largeOrEqual(b);
    return !(a.abs_large(b));

}


bool operator < (const BigInt& a, const BigInt& b)
{
    return !(a >= b);
}



bool operator <= (const BigInt& a, const BigInt& b)
{
    return !(a > b);
}


BigInt operator - (const BigInt& a, const BigInt& b)
{
    BigInt res;

    if(a.is_positive == b.is_positive)
    {
        if(a.abs_large(b))
        {
            res = a;
            res.abs_sub(b);
        }
        else
        {
            res = b;
            res.is_positive = !a.is_positive;
            res.abs_sub(a);
        }
    }
    else
    {
        res = a;
        res.abs_add(b);
    }

    return res;
}

BigInt operator + (const BigInt& a, const BigInt& b)
{
    BigInt res;

    if(a.is_positive == b.is_positive)
    {
        res = a;
        res.abs_add(b);
    }
    else
    {
        if(a.abs_large(b))
        {
            res = a;
            res.abs_sub(b);
        }
        else
        {
            res = b;
            res.abs_sub(a);
        }
    }

    return res;
}

BigInt operator << (const BigInt& a, const unsigned int b)
{
    BigInt res(a);
    return res.leftShiftInplace(b);
}

BigInt operator >> (const BigInt& a, const unsigned int b)
{
    BigInt res(a);
    return res.rightShiftInplace(b);
}

BigInt operator * (const BigInt& a, const BigInt& b)
{
    BigInt res(0, a.is_positive == b.is_positive);
    BigInt shift_int(b);

    int lastShift = 0;
    for(int i = 0; i < a.data_bit_num; i++)
    {
        if(a.at(i))
        {
            // cout<<"i:" <<i - lastShift<< endl;
            shift_int.leftShiftInplace(i - lastShift);
            // cout<<shift_int.data_bit_num<<endl;
            // cout<<"si:"<<shift_int.toHexStr()<<endl;
            // cout<<"shift done" << endl;
            res.abs_add(shift_int);
            // cout<<"add done"<<endl;
            lastShift = i;
        }
    }  
    // cout<<"All done"<<endl;
    res.count_bit_nums();
    return res;
    // if(a.data_bit_num > b.data_bit_num)
    // {

    //     int lastShift = 0;
    //     for(int i = 0; i < data_bit_num; i++)
    //     {
    //         if(flag.at(i))
    //         {
    //             shift_int.leftShiftInplace(i - lastShift);
    //             abs_add(shift_int);
    //             lastShift = i;
    //         }
    //     }
    // }
} 


BigInt operator % (const BigInt& a, const BigInt& b)
{
    if(a.data == b.data )
        return BigInt(0);
    else if(b.abs_large(a))
    {
        // cout<<"???"<<endl;
        return a;
    }
    BigInt res(a);
    BigInt quot(0);
    res.abs_mod(b, quot);

    return res;
}

BigInt operator / (const BigInt& a, const BigInt& b)
{
    BigInt res(0);
    BigInt temp(a);
    temp.abs_mod(b, res);
    res.is_positive = (a.is_positive == b.is_positive);
    return res;
}

BigInt BigInt::modexp(const BigInt& base, const BigInt& mod) const
{
    BigInt res(ONE);

    for(int i = base.data_bit_num - 1; i>=0; i--)
    {
        res = (res * res) % mod;
        if(base.at(i))
            res = (res*(*this))%mod;

    }
    return res;
}


//Todo: 优化
BigInt& BigInt::abs_mod(const BigInt& b,  BigInt& quot)
{
    int shift = 0;
    while(true)
    {
        shift = data_bit_num - b.data_bit_num;
       
        BigInt temp;
        while(shift >= 0)
        {
            temp = b << shift;
            if(!temp.abs_large(*this))
                break;
            shift--;
        }

        if(shift < 0)
            break;
        
        BigInt one = ONE << shift;
        abs_sub(temp);
        quot.abs_add(one);
    }

    return *this;
}

BigInt& BigInt:: rightShiftInplace(const unsigned int b)
{
    
    if(b > 0)
    {   
        if(b > data_bit_num)
         {
            *this = ZERO;
            return *this;
         }   

        unsigned int base_shift = b >> base_bit_expo;
        unsigned int offset = b & base_bit_remi;
        base_t data_to_next_baset = 0;
        unsigned int left_shift_num = base_bit_num - offset;
        int i = 0;
        for(; i + base_shift < data.size(); i++)
        {
            data[i] = data[i + base_shift];
        }
        
        while(i < data.size())
            data[i++] = 0;
        
        trim();
        data[0] >>= offset;
        for(int i = 1; i<data.size(); i++)
        {
            data_to_next_baset = data[i] << left_shift_num;
            data[i - 1] |= data_to_next_baset;
            data[i] >>= offset;
        }

        data_bit_num -= b;
        trim();
        
    }
    return *this;
}

BigInt& BigInt:: leftShiftInplace(const unsigned int b)
{
    if(b > 0)
    {
        int base_shift = b >> base_bit_expo;
        int offset = b & base_bit_remi;
        int orgin_size = data.size();
        // cout<<base_shift<<endl;
        for(int i = 0; i<base_shift; i++)
        {
            data.push_back(0);
        }

   
        int i = int(data.size()) - 1;
            //  cout<<i -base_shift<<endl;
        for(int j = i - base_shift; j>=0; j--, i--)
            data[i] = data[j];
        while(i>=0)
            data[i--] = 0;

        i = int(data.size()) - 1;
        base_t data_to_next_baset = 0;
        int right_shift_num = base_bit_num - offset;
        data.push_back(0);
        // cout<<orgin_size - 1<<endl;
        for(int j = orgin_size - 1; j>=0; j--, i--)
        {
            data_to_next_baset = data[i] >> right_shift_num;
            data[i + 1] |= data_to_next_baset;
            data[i] <<= offset;
        }

        data_bit_num += b;
        trim();
    }
    return *this;
}


// BigInt BigInt::abs_multi(const BigInt& b)
// {

// }

BigInt& BigInt::abs_add(const BigInt& b)
{
    if(b.data.size() > data.size())
    {
        int add = b.data.size() - data.size();
        for(int i = 0; i<add; i++)
        {
            data.push_back(0);
        }
    }

    base_t carry = 0;
    base_t middle_value = 0;
    int i = 0;
    for( ;i<b.data.size(); i++)
    {
        middle_value = data[i] + b.data[i] + carry;
        if(middle_value < data[i] + b.data[i] || middle_value < data[i] 
            || middle_value < b.data[i])
        {
            carry = 1;
        }
        else 
            carry = 0;
        data[i] = middle_value;
    }

    while (i<data.size() && carry > 0)
    {
        middle_value = data[i] + carry;
        if(middle_value < data[i] || middle_value < carry)
        {
            carry = 1;
           
        }
        else
        {
            carry = 0;
        }
        data[i] = middle_value;
    }
    
    if(carry)
    {
        data.push_back(1);
    }

    count_bit_nums();
    return *this;

}

// abs(b) <= abs(this)
BigInt& BigInt::abs_sub(const BigInt& b)
{
    base_t carry = 0;

    for(int i = 0; i<b.data.size(); i++)
    {
        if(b.data[i] > data[i] || b.data[i] == data[i] && carry > 0)
        {
            data[i] = base - b.data[i] - carry + data[i] + 1;
            carry = 1;
        }
        else
        {
            data[i] = data[i] - b.data[i] - carry;
            carry = 0;
        }
    }

    if(carry)
        data[b.data.size()] -= 1;

    trim();
    return *this;
}

void BigInt::count_bit_nums()
{
    if(data.size() == 1 && data.back() == 0)
    {
        data_bit_num = 1;
        return;
    }

    data_bit_num = data.size() << (base_bit_expo);
    base_t last = data[data.size() - 1];
    base_t shift_bit = base_t(1) << (base_bit_num - 1);

    while(!(shift_bit & last) && data_bit_num > 1)
    {
        shift_bit >>= 1;
        data_bit_num--;
    }
}
