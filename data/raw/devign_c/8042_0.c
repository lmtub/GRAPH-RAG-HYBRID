static int doTest(uint8_t *ref[4], int refStride[4], int w, int h,

                  enum PixelFormat srcFormat, enum PixelFormat dstFormat,

                  int srcW, int srcH, int dstW, int dstH, int flags)

{

    static enum PixelFormat cur_srcFormat;

    static int cur_srcW, cur_srcH;

    static uint8_t *src[4];

    static int srcStride[4];

    uint8_t *dst[4] = {0};

    uint8_t *out[4] = {0};

    int dstStride[4];

    int i;

    uint64_t ssdY, ssdU=0, ssdV=0, ssdA=0;

    struct SwsContext *dstContext = NULL, *outContext = NULL;

    uint32_t crc = 0;

    int res = 0;



    if (cur_srcFormat != srcFormat || cur_srcW != srcW || cur_srcH != srcH) {

        struct SwsContext *srcContext = NULL;

        int p;



        for (p = 0; p < 4; p++)

            if (src[p])

                av_freep(&src[p]);



        av_image_fill_linesizes(srcStride, srcFormat, srcW);

        for (p = 0; p < 4; p++) {

            if (srcStride[p])

                src[p] = av_mallocz(srcStride[p]*srcH+16);

            if (srcStride[p] && !src[p]) {

                perror("Malloc");

                res = -1;



                goto end;

            }

        }

        srcContext = sws_getContext(w, h, PIX_FMT_YUVA420P, srcW, srcH,

                                    srcFormat, SWS_BILINEAR, NULL, NULL, NULL);

        if (!srcContext) {

            fprintf(stderr, "Failed to get %s ---> %s\n",

                    av_pix_fmt_descriptors[PIX_FMT_YUVA420P].name,

                    av_pix_fmt_descriptors[srcFormat].name);

            res = -1;



            goto end;

        }

        sws_scale(srcContext, ref, refStride, 0, h, src, srcStride);

        sws_freeContext(srcContext);



        cur_srcFormat = srcFormat;

        cur_srcW = srcW;

        cur_srcH = srcH;

    }



    av_image_fill_linesizes(dstStride, dstFormat, dstW);

    for (i=0; i<4; i++) {

        /* Image buffers passed into libswscale can be allocated any way you

         * prefer, as long as they're aligned enough for the architecture, and

         * they're freed appropriately (such as using av_free for buffers

         * allocated with av_malloc). */

        /* An extra 16 bytes is being allocated because some scalers may write

         * out of bounds. */

        if (dstStride[i])

            dst[i]= av_mallocz(dstStride[i]*dstH+16);

        if (refStride[i])

            out[i]= av_mallocz(refStride[i]*h);

        if ((dstStride[i] && !dst[i]) || (refStride[i] && !out[i])) {

            perror("Malloc");

            res = -1;



            goto end;

        }

    }



    dstContext= sws_getContext(srcW, srcH, srcFormat, dstW, dstH, dstFormat, flags, NULL, NULL, NULL);

    if (!dstContext) {

        fprintf(stderr, "Failed to get %s ---> %s\n",

                av_pix_fmt_descriptors[srcFormat].name,

                av_pix_fmt_descriptors[dstFormat].name);

        res = -1;



        goto end;

    }

    outContext= sws_getContext(dstW, dstH, dstFormat, w, h, PIX_FMT_YUVA420P, flags, NULL, NULL, NULL);

    if (!outContext) {

        fprintf(stderr, "Failed to get %s ---> %s\n",

                av_pix_fmt_descriptors[dstFormat].name,

                av_pix_fmt_descriptors[PIX_FMT_YUVA420P].name);

        res = -1;



        goto end;

    }

//    printf("test %X %X %X -> %X %X %X\n", (int)ref[0], (int)ref[1], (int)ref[2],

//        (int)src[0], (int)src[1], (int)src[2]);



    printf(" %s %dx%d -> %s %3dx%3d flags=%2d",

           av_pix_fmt_descriptors[srcFormat].name, srcW, srcH,

           av_pix_fmt_descriptors[dstFormat].name, dstW, dstH,

           flags);

    fflush(stdout);



    sws_scale(dstContext, src, srcStride, 0, srcH, dst, dstStride);

    sws_scale(outContext, dst, dstStride, 0, dstH, out, refStride);



    for (i = 0; i < 4 && dstStride[i]; i++) {

        crc = av_crc(av_crc_get_table(AV_CRC_32_IEEE), crc, dst[i], dstStride[i] * dstH);

    }



    ssdY= getSSD(ref[0], out[0], refStride[0], refStride[0], w, h);

    if (hasChroma(srcFormat) && hasChroma(dstFormat)) {

        //FIXME check that output is really gray

        ssdU= getSSD(ref[1], out[1], refStride[1], refStride[1], (w+1)>>1, (h+1)>>1);

        ssdV= getSSD(ref[2], out[2], refStride[2], refStride[2], (w+1)>>1, (h+1)>>1);

    }

    if (isALPHA(srcFormat) && isALPHA(dstFormat))

        ssdA= getSSD(ref[3], out[3], refStride[3], refStride[3], w, h);



    ssdY/= w*h;

    ssdU/= w*h/4;

    ssdV/= w*h/4;

    ssdA/= w*h;



    printf(" CRC=%08x SSD=%5"PRId64",%5"PRId64",%5"PRId64",%5"PRId64"\n",

           crc, ssdY, ssdU, ssdV, ssdA);



end:



    sws_freeContext(dstContext);

    sws_freeContext(outContext);



    for (i=0; i<4; i++) {

        if (dstStride[i])

            av_free(dst[i]);

        if (refStride[i])

            av_free(out[i]);

    }



    return res;

}
