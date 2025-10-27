static int sort_stt(FFV1Context *s, uint8_t stt[256])

{

    int i, i2, changed, print = 0;



    do {

        changed = 0;

        for (i = 12; i < 244; i++) {

            for (i2 = i + 1; i2 < 245 && i2 < i + 4; i2++) {



#define COST(old, new)                                      \

    s->rc_stat[old][0] * -log2((256 - (new)) / 256.0) +     \

    s->rc_stat[old][1] * -log2((new)         / 256.0)



#define COST2(old, new)                         \

    COST(old, new) + COST(256 - (old), 256 - (new))



                double size0 = COST2(i,  i) + COST2(i2, i2);

                double sizeX = COST2(i, i2) + COST2(i2, i);

                if (sizeX < size0 && i != 128 && i2 != 128) {

                    int j;

                    FFSWAP(int, stt[i], stt[i2]);

                    FFSWAP(int, s->rc_stat[i][0], s->rc_stat[i2][0]);

                    FFSWAP(int, s->rc_stat[i][1], s->rc_stat[i2][1]);

                    if (i != 256 - i2) {

                        FFSWAP(int, stt[256 - i], stt[256 - i2]);

                        FFSWAP(int, s->rc_stat[256 - i][0], s->rc_stat[256 - i2][0]);

                        FFSWAP(int, s->rc_stat[256 - i][1], s->rc_stat[256 - i2][1]);

                    }

                    for (j = 1; j < 256; j++) {

                        if (stt[j] == i)

                            stt[j] = i2;

                        else if (stt[j] == i2)

                            stt[j] = i;

                        if (i != 256 - i2) {

                            if (stt[256 - j] == 256 - i)

                                stt[256 - j] = 256 - i2;

                            else if (stt[256 - j] == 256 - i2)

                                stt[256 - j] = 256 - i;

                        }

                    }

                    print = changed = 1;

                }

            }

        }

    } while (changed);

    return print;

}
