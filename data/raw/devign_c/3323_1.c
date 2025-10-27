int64_t av_gcd(int64_t a, int64_t b)

{

    if (b)

        return av_gcd(b, a % b);

    else

        return a;

}
