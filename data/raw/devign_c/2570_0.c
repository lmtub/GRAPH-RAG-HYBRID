static unsigned int rms(const int *data)

{

    int x;

    unsigned int res = 0x10000;

    int b = 0;



    for (x=0; x<10; x++) {

        res = (((0x1000000 - (*data) * (*data)) >> 12) * res) >> 12;



        if (res == 0)

            return 0;



        while (res <= 0x3fff) {

            b++;

            res <<= 2;

        }

        data++;

    }



    if (res > 0)

        res = t_sqrt(res);



    res >>= (b + 10);

    return res;

}
