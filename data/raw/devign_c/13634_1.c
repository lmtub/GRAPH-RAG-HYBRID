static int quant_psnr8x8_c(MpegEncContext *s, uint8_t *src1,

                           uint8_t *src2, ptrdiff_t stride, int h)

{

    LOCAL_ALIGNED_16(int16_t, temp, [64 * 2]);

    int16_t *const bak = temp + 64;

    int sum = 0, i;



    av_assert2(h == 8);

    s->mb_intra = 0;



    s->pdsp.diff_pixels(temp, src1, src2, stride);



    memcpy(bak, temp, 64 * sizeof(int16_t));



    s->block_last_index[0 /* FIXME */] =

        s->fast_dct_quantize(s, temp, 0 /* FIXME */, s->qscale, &i);

    s->dct_unquantize_inter(s, temp, 0, s->qscale);

    ff_simple_idct_8(temp); // FIXME



    for (i = 0; i < 64; i++)

        sum += (temp[i] - bak[i]) * (temp[i] - bak[i]);



    return sum;

}
