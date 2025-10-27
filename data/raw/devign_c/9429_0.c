int ff_sws_alphablendaway(SwsContext *c, const uint8_t *src[],

                          int srcStride[], int srcSliceY, int srcSliceH,

                          uint8_t *dst[], int dstStride[])

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(c->srcFormat);

    int nb_components = desc->nb_components;

    int plane, x, y;

    int plane_count = isGray(c->srcFormat) ? 1 : 3;

    int sixteen_bits = desc->comp[0].depth_minus1 >= 8;

    unsigned off    = 1<<desc->comp[0].depth_minus1;

    unsigned shift  = desc->comp[0].depth_minus1 + 1;

    unsigned max    = (1<<shift) - 1;



    av_assert0(plane_count == nb_components - 1);

    if (desc->flags & AV_PIX_FMT_FLAG_PLANAR) {

        for (plane = 0; plane < plane_count; plane++) {

            int w = plane ? c->chrSrcW : c->srcW;

            int y_subsample = plane ? desc->log2_chroma_h: 0;

            for (y = srcSliceY >> y_subsample; y < FF_CEIL_RSHIFT(srcSliceH, y_subsample); y++) {

                if (sixteen_bits) {

                    const uint16_t *s = src[plane      ] + srcStride[plane] * y;

                    const uint16_t *a = src[plane_count] + srcStride[plane_count] * y;

                          uint16_t *d = dst[plane      ] + dstStride[plane] * y;

                    unsigned target = plane && !(desc->flags & AV_PIX_FMT_FLAG_RGB) ? 1<<desc->comp[0].depth_minus1 : 0;

                    if ((!isBE(c->srcFormat)) == !HAVE_BIGENDIAN) {

                        for (x = 0; x < w; x++) {

                            unsigned u = s[x]*a[x] + target*(max-a[x]) + off;

                            d[x] = av_clip((u + (u >> shift)) >> shift, 0, max);

                        }

                    } else {

                        for (x = 0; x < w; x++) {

                            unsigned aswap =av_bswap16(a[x]);

                            unsigned u = av_bswap16(s[x])*aswap + target*(max-aswap) + off;

                            d[x] = av_clip((u + (u >> shift)) >> shift, 0, max);

                        }

                    }

                } else {

                    const uint8_t *s = src[plane      ] + srcStride[plane] * y;

                    const uint8_t *a = src[plane_count] + srcStride[plane_count] * y;

                          uint8_t *d = dst[plane      ] + dstStride[plane] * y;

                    unsigned target = plane && !(desc->flags & AV_PIX_FMT_FLAG_RGB) ? 128 : 0;

                    for (x = 0; x < w; x++) {

                        unsigned u = s[x]*a[x] + target*(255-a[x]) + 128;

                        d[x] = (257*u) >> 16;

                    }

                }

            }

        }

    } else {

        int alpha_pos = desc->comp[plane_count].offset_plus1 - 1;

        int w = c->srcW;

        for (y = srcSliceY; y < srcSliceH; y++) {

            if (sixteen_bits) {

                const uint16_t *s = src[0] + srcStride[0] * y + 2*!alpha_pos;

                const uint16_t *a = src[0] + srcStride[0] * y + alpha_pos;

                        uint16_t *d = dst[0] + dstStride[0] * y;

                if ((!isBE(c->srcFormat)) == !HAVE_BIGENDIAN) {

                    for (x = 0; x < w; x++) {

                        for (plane = 0; plane < plane_count; plane++) {

                            unsigned target = plane && !(desc->flags & AV_PIX_FMT_FLAG_RGB) ? 1<<desc->comp[0].depth_minus1 : 0;

                            int x_index = (plane_count + 1) * x;

                            unsigned u = s[x_index + plane]*a[x_index] + target*(max-a[x_index]) + off;

                            d[plane_count*x + plane] = av_clip((u + (u >> shift)) >> shift, 0, max);

                        }

                    }

                } else {

                    for (x = 0; x < w; x++) {

                        for (plane = 0; plane < plane_count; plane++) {

                            unsigned target = plane && !(desc->flags & AV_PIX_FMT_FLAG_RGB) ? 1<<desc->comp[0].depth_minus1 : 0;

                            int x_index = (plane_count + 1) * x;

                            unsigned aswap =av_bswap16(a[x_index]);

                            unsigned u = av_bswap16(s[x_index + plane])*aswap + target*(max-aswap) + off;

                            d[plane_count*x + plane] = av_clip((u + (u >> shift)) >> shift, 0, max);

                        }

                    }

                }

            } else {

                const uint8_t *s = src[0] + srcStride[0] * y + !alpha_pos;

                const uint8_t *a = src[0] + srcStride[0] * y + alpha_pos;

                      uint8_t *d = dst[0] + dstStride[0] * y;

                for (x = 0; x < w; x++) {

                    for (plane = 0; plane < plane_count; plane++) {

                        unsigned target = plane && !(desc->flags & AV_PIX_FMT_FLAG_RGB) ? 128 : 0;

                        int x_index = (plane_count + 1) * x;

                        unsigned u = s[x_index + plane]*a[x_index] + target*(255-a[x_index]) + 128;

                        d[plane_count*x + plane] = (257*u) >> 16;

                    }

                }

            }

        }

    }



    return 0;

}
