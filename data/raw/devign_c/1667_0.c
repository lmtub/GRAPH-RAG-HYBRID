static void zero_remaining(unsigned int b, unsigned int b_max,

                           const unsigned int *div_blocks, int32_t *buf)

{

    unsigned int count = 0;



    while (b < b_max)

        count += div_blocks[b];



    if (count)

        memset(buf, 0, sizeof(*buf) * count);

}
