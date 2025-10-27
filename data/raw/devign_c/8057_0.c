int av_reduce(int *dst_num, int *dst_den, int64_t num, int64_t den, int64_t max){

    AVRational a0={0,1}, a1={1,0};

    int sign= (num<0) ^ (den<0);

    int64_t gcd= av_gcd(FFABS(num), FFABS(den));



    if(gcd){

        num = FFABS(num)/gcd;

        den = FFABS(den)/gcd;

    }

    if(num<=max && den<=max){

        a1= (AVRational){num, den};

        den=0;

    }



    while(den){

        uint64_t x      = num / den;

        int64_t next_den= num - den*x;

        int64_t a2n= x*a1.num + a0.num;

        int64_t a2d= x*a1.den + a0.den;



        if(a2n > max || a2d > max){

            if(a1.num) x= (max - a0.num) / a1.num;

            if(a1.den) x= FFMIN(x, (max - a0.den) / a1.den);



            if (den*(2*x*a1.den + a0.den) > num*a1.den)

                a1 = (AVRational){x*a1.num + a0.num, x*a1.den + a0.den};

            break;

        }



        a0= a1;

        a1= (AVRational){a2n, a2d};

        num= den;

        den= next_den;

    }

    assert(av_gcd(a1.num, a1.den) <= 1U);



    *dst_num = sign ? -a1.num : a1.num;

    *dst_den = a1.den;



    return den==0;

}
