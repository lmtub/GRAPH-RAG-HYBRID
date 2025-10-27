int attribute_align_arg sws_scale(struct SwsContext *c,

                                  const uint8_t * const srcSlice[],

                                  const int srcStride[], int srcSliceY,

                                  int srcSliceH, uint8_t *const dst[],

                                  const int dstStride[])

{

    int i, ret;

    const uint8_t *src2[4] = { srcSlice[0], srcSlice[1], srcSlice[2], srcSlice[3] };

    uint8_t *dst2[4] = { dst[0], dst[1], dst[2], dst[3] };

    uint8_t *rgb0_tmp = NULL;



    // do not mess up sliceDir if we have a "trailing" 0-size slice

    if (srcSliceH == 0)

        return 0;



    if (!check_image_pointers(srcSlice, c->srcFormat, srcStride)) {

        av_log(c, AV_LOG_ERROR, "bad src image pointers\n");

        return 0;

    }

    if (!check_image_pointers((const uint8_t* const*)dst, c->dstFormat, dstStride)) {

        av_log(c, AV_LOG_ERROR, "bad dst image pointers\n");

        return 0;

    }



    if (c->sliceDir == 0 && srcSliceY != 0 && srcSliceY + srcSliceH != c->srcH) {

        av_log(c, AV_LOG_ERROR, "Slices start in the middle!\n");

        return 0;

    }

    if (c->sliceDir == 0) {

        if (srcSliceY == 0) c->sliceDir = 1; else c->sliceDir = -1;

    }



    if (usePal(c->srcFormat)) {

        for (i = 0; i < 256; i++) {

            int p, r, g, b, y, u, v, a = 0xff;

            if (c->srcFormat == AV_PIX_FMT_PAL8) {

                p = ((const uint32_t *)(srcSlice[1]))[i];

                a = (p >> 24) & 0xFF;

                r = (p >> 16) & 0xFF;

                g = (p >>  8) & 0xFF;

                b =  p        & 0xFF;

            } else if (c->srcFormat == AV_PIX_FMT_RGB8) {

                r = ( i >> 5     ) * 36;

                g = ((i >> 2) & 7) * 36;

                b = ( i       & 3) * 85;

            } else if (c->srcFormat == AV_PIX_FMT_BGR8) {

                b = ( i >> 6     ) * 85;

                g = ((i >> 3) & 7) * 36;

                r = ( i       & 7) * 36;

            } else if (c->srcFormat == AV_PIX_FMT_RGB4_BYTE) {

                r = ( i >> 3     ) * 255;

                g = ((i >> 1) & 3) * 85;

                b = ( i       & 1) * 255;

            } else if (c->srcFormat == AV_PIX_FMT_GRAY8 || c->srcFormat == AV_PIX_FMT_GRAY8A) {

                r = g = b = i;

            } else {

                av_assert1(c->srcFormat == AV_PIX_FMT_BGR4_BYTE);

                b = ( i >> 3     ) * 255;

                g = ((i >> 1) & 3) * 85;

                r = ( i       & 1) * 255;

            }

#define RGB2YUV_SHIFT 15

#define BY ( (int) (0.114 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define BV (-(int) (0.081 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define BU ( (int) (0.500 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define GY ( (int) (0.587 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define GV (-(int) (0.419 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define GU (-(int) (0.331 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define RY ( (int) (0.299 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define RV ( (int) (0.500 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

#define RU (-(int) (0.169 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))



            y = av_clip_uint8((RY * r + GY * g + BY * b + ( 33 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);

            u = av_clip_uint8((RU * r + GU * g + BU * b + (257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);

            v = av_clip_uint8((RV * r + GV * g + BV * b + (257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);

            c->pal_yuv[i]= y + (u<<8) + (v<<16) + ((unsigned)a<<24);



            switch (c->dstFormat) {

            case AV_PIX_FMT_BGR32:

#if !HAVE_BIGENDIAN

            case AV_PIX_FMT_RGB24:

#endif

                c->pal_rgb[i]=  r + (g<<8) + (b<<16) + ((unsigned)a<<24);

                break;

            case AV_PIX_FMT_BGR32_1:

#if HAVE_BIGENDIAN

            case AV_PIX_FMT_BGR24:

#endif

                c->pal_rgb[i]= a + (r<<8) + (g<<16) + ((unsigned)b<<24);

                break;

            case AV_PIX_FMT_RGB32_1:

#if HAVE_BIGENDIAN

            case AV_PIX_FMT_RGB24:

#endif

                c->pal_rgb[i]= a + (b<<8) + (g<<16) + ((unsigned)r<<24);

                break;

            case AV_PIX_FMT_RGB32:

#if !HAVE_BIGENDIAN

            case AV_PIX_FMT_BGR24:

#endif

            default:

                c->pal_rgb[i]=  b + (g<<8) + (r<<16) + ((unsigned)a<<24);

            }

        }

    }



    if (c->src0Alpha && !c->dst0Alpha && isALPHA(c->dstFormat)) {

        uint8_t *base;

        int x,y;

        rgb0_tmp = av_malloc(FFABS(srcStride[0]) * srcSliceH + 32);

        base = srcStride[0] < 0 ? rgb0_tmp - srcStride[0] * (srcSliceH-1) : rgb0_tmp;

        for (y=0; y<srcSliceH; y++){

            memcpy(base + srcStride[0]*y, src2[0] + srcStride[0]*y, 4*c->srcW);

            for (x=c->src0Alpha-1; x<4*c->srcW; x+=4) {

                base[ srcStride[0]*y + x] = 0xFF;

            }

        }

        src2[0] = base;

    }



    // copy strides, so they can safely be modified

    if (c->sliceDir == 1) {

        // slices go from top to bottom

        int srcStride2[4] = { srcStride[0], srcStride[1], srcStride[2],

                              srcStride[3] };

        int dstStride2[4] = { dstStride[0], dstStride[1], dstStride[2],

                              dstStride[3] };



        reset_ptr(src2, c->srcFormat);

        reset_ptr((void*)dst2, c->dstFormat);



        /* reset slice direction at end of frame */

        if (srcSliceY + srcSliceH == c->srcH)

            c->sliceDir = 0;



        ret = c->swScale(c, src2, srcStride2, srcSliceY, srcSliceH, dst2,

                          dstStride2);

    } else {

        // slices go from bottom to top => we flip the image internally

        int srcStride2[4] = { -srcStride[0], -srcStride[1], -srcStride[2],

                              -srcStride[3] };

        int dstStride2[4] = { -dstStride[0], -dstStride[1], -dstStride[2],

                              -dstStride[3] };



        src2[0] += (srcSliceH - 1) * srcStride[0];

        if (!usePal(c->srcFormat))

            src2[1] += ((srcSliceH >> c->chrSrcVSubSample) - 1) * srcStride[1];

        src2[2] += ((srcSliceH >> c->chrSrcVSubSample) - 1) * srcStride[2];

        src2[3] += (srcSliceH - 1) * srcStride[3];

        dst2[0] += ( c->dstH                         - 1) * dstStride[0];

        dst2[1] += ((c->dstH >> c->chrDstVSubSample) - 1) * dstStride[1];

        dst2[2] += ((c->dstH >> c->chrDstVSubSample) - 1) * dstStride[2];

        dst2[3] += ( c->dstH                         - 1) * dstStride[3];



        reset_ptr(src2, c->srcFormat);

        reset_ptr((void*)dst2, c->dstFormat);



        /* reset slice direction at end of frame */

        if (!srcSliceY)

            c->sliceDir = 0;



        ret = c->swScale(c, src2, srcStride2, c->srcH-srcSliceY-srcSliceH,

                          srcSliceH, dst2, dstStride2);

    }



    av_free(rgb0_tmp);

    return ret;

}
