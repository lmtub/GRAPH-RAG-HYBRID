static int bit8x8_c(MpegEncContext *s, uint8_t *src1, uint8_t *src2,

                    ptrdiff_t stride, int h)

{

    const uint8_t *scantable = s->intra_scantable.permutated;

    LOCAL_ALIGNED_16(int16_t, temp, [64]);

    int i, last, run, bits, level, start_i;

    const int esc_length = s->ac_esc_length;

    uint8_t *length, *last_length;



    av_assert2(h == 8);



    s->pdsp.diff_pixels(temp, src1, src2, stride);



    s->block_last_index[0 /* FIXME */] =

    last                               =

        s->fast_dct_quantize(s, temp, 0 /* FIXME */, s->qscale, &i);



    bits = 0;



    if (s->mb_intra) {

        start_i     = 1;

        length      = s->intra_ac_vlc_length;

        last_length = s->intra_ac_vlc_last_length;

        bits       += s->luma_dc_vlc_length[temp[0] + 256]; // FIXME: chroma

    } else {

        start_i     = 0;

        length      = s->inter_ac_vlc_length;

        last_length = s->inter_ac_vlc_last_length;

    }



    if (last >= start_i) {

        run = 0;

        for (i = start_i; i < last; i++) {

            int j = scantable[i];

            level = temp[j];



            if (level) {

                level += 64;

                if ((level & (~127)) == 0)

                    bits += length[UNI_AC_ENC_INDEX(run, level)];

                else

                    bits += esc_length;

                run = 0;

            } else

                run++;

        }

        i = scantable[last];



        level = temp[i] + 64;



        av_assert2(level - 64);



        if ((level & (~127)) == 0)

            bits += last_length[UNI_AC_ENC_INDEX(run, level)];

        else

            bits += esc_length;

    }



    return bits;

}
