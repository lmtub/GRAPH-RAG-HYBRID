static int packed_vscale(SwsContext *c, SwsFilterDescriptor *desc, int sliceY, int sliceH)

{

    VScalerContext *inst = desc->instance;

    int dstW = desc->dst->width;

    int chrSliceY = sliceY >> desc->dst->v_chr_sub_sample;



    int lum_fsize = inst[0].filter_size;

    int chr_fsize = inst[1].filter_size;

    uint16_t *lum_filter = inst[0].filter[0];

    uint16_t *chr_filter = inst[1].filter[0];



    int firstLum = FFMAX(1-lum_fsize, inst[0].filter_pos[chrSliceY]);

    int firstChr = FFMAX(1-chr_fsize, inst[1].filter_pos[chrSliceY]);



    int sp0 = firstLum - desc->src->plane[0].sliceY;

    int sp1 = firstChr - desc->src->plane[1].sliceY;

    int sp2 = firstChr - desc->src->plane[2].sliceY;

    int sp3 = firstLum - desc->src->plane[3].sliceY;

    int dp = sliceY - desc->dst->plane[0].sliceY;

    uint8_t **src0 = desc->src->plane[0].line + sp0;

    uint8_t **src1 = desc->src->plane[1].line + sp1;

    uint8_t **src2 = desc->src->plane[2].line + sp2;

    uint8_t **src3 = desc->alpha ? desc->src->plane[3].line + sp3 : NULL;

    uint8_t **dst = desc->dst->plane[0].line + dp;





    if (c->yuv2packed1 && lum_fsize == 1 && chr_fsize <= 2) { // unscaled RGB

        int chrAlpha = chr_fsize == 1 ? 0 : chr_filter[2 * sliceY + 1];

        ((yuv2packed1_fn)inst->pfn)(c, (const int16_t*)*src0, (const int16_t**)src1, (const int16_t**)src2, (const int16_t*)(desc->alpha ? *src3 : NULL),  *dst, dstW, chrAlpha, sliceY);

    } else if (c->yuv2packed2 && lum_fsize == 2 && chr_fsize == 2) { // bilinear upscale RGB

        int lumAlpha = lum_filter[2 * sliceY + 1];

        int chrAlpha = chr_filter[2 * sliceY + 1];

        c->lumMmxFilter[2] =

        c->lumMmxFilter[3] = lum_filter[2 * sliceY]    * 0x10001;

        c->chrMmxFilter[2] =

        c->chrMmxFilter[3] = chr_filter[2 * chrSliceY] * 0x10001;

        ((yuv2packed2_fn)inst->pfn)(c, (const int16_t**)src0, (const int16_t**)src1, (const int16_t**)src2, (const int16_t**)src3,

                    *dst, dstW, lumAlpha, chrAlpha, sliceY);

    } else { // general RGB

        ((yuv2packedX_fn)inst->pfn)(c, lum_filter + sliceY * lum_fsize,

                    (const int16_t**)src0, lum_fsize, chr_filter + sliceY * chr_fsize,

                    (const int16_t**)src1, (const int16_t**)src2, chr_fsize, (const int16_t**)src3, *dst, dstW, sliceY);

    }

    return 1;

}
