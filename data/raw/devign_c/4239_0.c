static int normalize_bits(int num, int width)

{

    int i = 0;

    int bits = (width) ? 31 : 15;

    int limit = 1 << (bits - 1);



    if (num) {

        if (num == -1)

            return bits;

        if (num < 0)

            num = ~num;

        for (i = 0; num < limit; i++)

            num <<= 1;

    }

    return i;

}
