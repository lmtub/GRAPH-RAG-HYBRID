static void decode(Real288_internal *glob, float gain, int cb_coef)

{

    unsigned int x, y;

    float f;

    double sum, sumsum;

    float *p1, *p2;

    float buffer[5];



    for (x=36; x--; glob->sb[x+5] = glob->sb[x]);



    for (x=5; x--;) {

        p1 = glob->sb+x;

        p2 = glob->pr1;

        for (sum=0, y=36; y--; sum -= (*(++p1))*(*(p2++)));



        glob->sb[x] = sum;

    }



    /* convert log and do rms */

    for (sum=32, x=10; x--; sum -= glob->pr2[x] * glob->lhist[x]);



    if (sum < 0)

        sum = 0;

    else if (sum > 60)

        sum = 60;



    sumsum = exp(sum * 0.1151292546497) * gain;    /* pow(10.0,sum/20)*f */



    for (sum=0, x=5; x--;) {

        buffer[x] = codetable[cb_coef][x] * sumsum;

        sum += buffer[x] * buffer[x];

    }



    if ((sum /= 5) < 1)

        sum = 1;



    /* shift and store */

    for (x=10; --x; glob->lhist[x] = glob->lhist[x-1]);



    *glob->lhist = glob->history[glob->phase] = 10 * log10(sum) - 32;



    for (x=1; x < 5; x++)

        for (y=x; y--; buffer[x] -= glob->pr1[x-y-1] * buffer[y]);



    /* output */

    for (x=0; x < 5; x++) {

        f = glob->sb[4-x] + buffer[x];



        if (f > 4095)

            f = 4095;

        else if (f < -4095)

            f = -4095;



        glob->output[glob->phasep+x] = glob->sb[4-x] = f;

    }

}
