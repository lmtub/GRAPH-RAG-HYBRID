static int doTest(uint8_t *ref[3], int refStride[3], int w, int h, int srcFormat, int dstFormat,

                  int srcW, int srcH, int dstW, int dstH, int flags){

    uint8_t *src[3];

    uint8_t *dst[3];

    uint8_t *out[3];

    int srcStride[3], dstStride[3];

    int i;

    uint64_t ssdY, ssdU, ssdV;

    struct SwsContext *srcContext, *dstContext, *outContext;

    int res;



    res = 0;

    for (i=0; i<3; i++){

        // avoid stride % bpp != 0

        if (srcFormat==PIX_FMT_RGB24 || srcFormat==PIX_FMT_BGR24)

            srcStride[i]= srcW*3;

        else

            srcStride[i]= srcW*4;



        if (dstFormat==PIX_FMT_RGB24 || dstFormat==PIX_FMT_BGR24)

            dstStride[i]= dstW*3;

        else

            dstStride[i]= dstW*4;



        src[i]= (uint8_t*) malloc(srcStride[i]*srcH);

        dst[i]= (uint8_t*) malloc(dstStride[i]*dstH);

        out[i]= (uint8_t*) malloc(refStride[i]*h);

        if (!src[i] || !dst[i] || !out[i]) {

            perror("Malloc");

            res = -1;



            goto end;

        }

    }



    dstContext = outContext = NULL;

    srcContext= sws_getContext(w, h, PIX_FMT_YUV420P, srcW, srcH, srcFormat, flags, NULL, NULL, NULL);

    if (!srcContext) {

        fprintf(stderr, "Failed to get %s ---> %s\n",

                sws_format_name(PIX_FMT_YUV420P),

                sws_format_name(srcFormat));

        res = -1;



        goto end;

    }

    dstContext= sws_getContext(srcW, srcH, srcFormat, dstW, dstH, dstFormat, flags, NULL, NULL, NULL);

    if (!dstContext) {

        fprintf(stderr, "Failed to get %s ---> %s\n",

                sws_format_name(srcFormat),

                sws_format_name(dstFormat));

        res = -1;



        goto end;

    }

    outContext= sws_getContext(dstW, dstH, dstFormat, w, h, PIX_FMT_YUV420P, flags, NULL, NULL, NULL);

    if (!outContext) {

        fprintf(stderr, "Failed to get %s ---> %s\n",

                sws_format_name(dstFormat),

                sws_format_name(PIX_FMT_YUV420P));

        res = -1;



        goto end;

    }

//    printf("test %X %X %X -> %X %X %X\n", (int)ref[0], (int)ref[1], (int)ref[2],

//        (int)src[0], (int)src[1], (int)src[2]);



    sws_scale(srcContext, ref, refStride, 0, h   , src, srcStride);

    sws_scale(dstContext, src, srcStride, 0, srcH, dst, dstStride);

    sws_scale(outContext, dst, dstStride, 0, dstH, out, refStride);



    ssdY= getSSD(ref[0], out[0], refStride[0], refStride[0], w, h);

    ssdU= getSSD(ref[1], out[1], refStride[1], refStride[1], (w+1)>>1, (h+1)>>1);

    ssdV= getSSD(ref[2], out[2], refStride[2], refStride[2], (w+1)>>1, (h+1)>>1);



    if (srcFormat == PIX_FMT_GRAY8 || dstFormat==PIX_FMT_GRAY8) ssdU=ssdV=0; //FIXME check that output is really gray



    ssdY/= w*h;

    ssdU/= w*h/4;

    ssdV/= w*h/4;



    printf(" %s %dx%d -> %s %4dx%4d flags=%2d SSD=%5lld,%5lld,%5lld\n",

           sws_format_name(srcFormat), srcW, srcH,

           sws_format_name(dstFormat), dstW, dstH,

           flags, ssdY, ssdU, ssdV);

    fflush(stdout);



    end:



    sws_freeContext(srcContext);

    sws_freeContext(dstContext);

    sws_freeContext(outContext);



    for (i=0; i<3; i++){

        free(src[i]);

        free(dst[i]);

        free(out[i]);

    }



    return res;

}
