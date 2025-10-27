static inline void mpeg4_encode_dc(PutBitContext * s, int level, int n)

{

#if 1


    level+=256;

    if (n < 4) {

        /* luminance */

        put_bits(s, uni_DCtab_lum_len[level], uni_DCtab_lum_bits[level]);

    } else {

        /* chrominance */

        put_bits(s, uni_DCtab_chrom_len[level], uni_DCtab_chrom_bits[level]);

    }

#else

    int size, v;

    /* find number of bits */

    size = 0;

    v = abs(level);

    while (v) {

        v >>= 1;

        size++;

    }



    if (n < 4) {

        /* luminance */

        put_bits(&s->pb, DCtab_lum[size][1], DCtab_lum[size][0]);

    } else {

        /* chrominance */

        put_bits(&s->pb, DCtab_chrom[size][1], DCtab_chrom[size][0]);

    }



    /* encode remaining bits */

    if (size > 0) {

        if (level < 0)

            level = (-level) ^ ((1 << size) - 1);

        put_bits(&s->pb, size, level);

        if (size > 8)

            put_bits(&s->pb, 1, 1);

    }

#endif

}