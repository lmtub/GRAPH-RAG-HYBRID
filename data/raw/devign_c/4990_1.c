static inline int64_t bs_get_v(const uint8_t **bs)

{

    int64_t v = 0;

    int br = 0;

    int c;



    do {

        c = **bs; (*bs)++;

        v <<= 7;

        v |= c & 0x7F;

        br++;

        if (br > 10)

            return -1;

    } while (c & 0x80);



    return v - br;

}
