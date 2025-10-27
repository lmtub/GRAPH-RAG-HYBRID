static inline int16_t calc_lowcomp(int16_t a, int16_t b0, int16_t b1, uint8_t bin)

{

    if (bin < 7) {

        if ((b0 + 256) == b1)

            a = 384;

        else if (b0 > b1)

            a = FFMAX(0, a - 64);

    }

    else if (bin < 20) {

        if ((b0 + 256) == b1)

            a = 320;

        else if (b0 > b1)

            a = FFMAX(0, a - 64);

    }

    else {

        a = FFMAX(0, a - 128);

    }



    return a;

}
