static void prodsum(float *tgt, float *src, int len, int n)

{

    unsigned int x;

    float *p1, *p2;

    double sum;



    while (n >= 0) {

        p1 = (p2 = src) - n;

        for (sum=0, x=len; x--; sum += (*p1++) * (*p2++));

        tgt[n--] = sum;

    }

}
