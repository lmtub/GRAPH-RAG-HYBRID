static int swScale(SwsContext *c, const uint8_t* src[],

                   int srcStride[], int srcSliceY,

                   int srcSliceH, uint8_t* dst[], int dstStride[])

{

    /* load a few things into local vars to make the code more readable? and faster */

    const int srcW= c->srcW;

    const int dstW= c->dstW;

    const int dstH= c->dstH;

    const int chrDstW= c->chrDstW;

    const int chrSrcW= c->chrSrcW;

    const int lumXInc= c->lumXInc;

    const int chrXInc= c->chrXInc;

    const enum PixelFormat dstFormat= c->dstFormat;

    const int flags= c->flags;

    int16_t *vLumFilterPos= c->vLumFilterPos;

    int16_t *vChrFilterPos= c->vChrFilterPos;

    int16_t *hLumFilterPos= c->hLumFilterPos;

    int16_t *hChrFilterPos= c->hChrFilterPos;

    int16_t *vLumFilter= c->vLumFilter;

    int16_t *vChrFilter= c->vChrFilter;

    int16_t *hLumFilter= c->hLumFilter;

    int16_t *hChrFilter= c->hChrFilter;

    int32_t *lumMmxFilter= c->lumMmxFilter;

    int32_t *chrMmxFilter= c->chrMmxFilter;

    const int vLumFilterSize= c->vLumFilterSize;

    const int vChrFilterSize= c->vChrFilterSize;

    const int hLumFilterSize= c->hLumFilterSize;

    const int hChrFilterSize= c->hChrFilterSize;

    int16_t **lumPixBuf= c->lumPixBuf;

    int16_t **chrUPixBuf= c->chrUPixBuf;

    int16_t **chrVPixBuf= c->chrVPixBuf;

    int16_t **alpPixBuf= c->alpPixBuf;

    const int vLumBufSize= c->vLumBufSize;

    const int vChrBufSize= c->vChrBufSize;

    uint8_t *formatConvBuffer= c->formatConvBuffer;

    const int chrSrcSliceY= srcSliceY >> c->chrSrcVSubSample;

    const int chrSrcSliceH= -((-srcSliceH) >> c->chrSrcVSubSample);

    int lastDstY;

    uint32_t *pal=c->pal_yuv;

    yuv2planar1_fn yuv2plane1 = c->yuv2plane1;

    yuv2planarX_fn yuv2planeX = c->yuv2planeX;

    yuv2interleavedX_fn yuv2nv12cX = c->yuv2nv12cX;

    yuv2packed1_fn yuv2packed1 = c->yuv2packed1;

    yuv2packed2_fn yuv2packed2 = c->yuv2packed2;

    yuv2packedX_fn yuv2packedX = c->yuv2packedX;

    int should_dither = is9_OR_10BPS(c->srcFormat) || is16BPS(c->srcFormat);



    /* vars which will change and which we need to store back in the context */

    int dstY= c->dstY;

    int lumBufIndex= c->lumBufIndex;

    int chrBufIndex= c->chrBufIndex;

    int lastInLumBuf= c->lastInLumBuf;

    int lastInChrBuf= c->lastInChrBuf;



    if (isPacked(c->srcFormat)) {

        src[0]=

        src[1]=

        src[2]=

        src[3]= src[0];

        srcStride[0]=

        srcStride[1]=

        srcStride[2]=

        srcStride[3]= srcStride[0];

    }

    srcStride[1]<<= c->vChrDrop;

    srcStride[2]<<= c->vChrDrop;



    DEBUG_BUFFERS("swScale() %p[%d] %p[%d] %p[%d] %p[%d] -> %p[%d] %p[%d] %p[%d] %p[%d]\n",

                  src[0], srcStride[0], src[1], srcStride[1], src[2], srcStride[2], src[3], srcStride[3],

                  dst[0], dstStride[0], dst[1], dstStride[1], dst[2], dstStride[2], dst[3], dstStride[3]);

    DEBUG_BUFFERS("srcSliceY: %d srcSliceH: %d dstY: %d dstH: %d\n",

                   srcSliceY,    srcSliceH,    dstY,    dstH);

    DEBUG_BUFFERS("vLumFilterSize: %d vLumBufSize: %d vChrFilterSize: %d vChrBufSize: %d\n",

                   vLumFilterSize,    vLumBufSize,    vChrFilterSize,    vChrBufSize);



    if (dstStride[0]%8 !=0 || dstStride[1]%8 !=0 || dstStride[2]%8 !=0 || dstStride[3]%8 != 0) {

        static int warnedAlready=0; //FIXME move this into the context perhaps

        if (flags & SWS_PRINT_INFO && !warnedAlready) {

            av_log(c, AV_LOG_WARNING, "Warning: dstStride is not aligned!\n"

                   "         ->cannot do aligned memory accesses anymore\n");

            warnedAlready=1;

        }

    }



    /* Note the user might start scaling the picture in the middle so this

       will not get executed. This is not really intended but works

       currently, so people might do it. */

    if (srcSliceY ==0) {

        lumBufIndex=-1;

        chrBufIndex=-1;

        dstY=0;

        lastInLumBuf= -1;

        lastInChrBuf= -1;

    }



    if (!should_dither) {

        c->chrDither8 = c->lumDither8 = ff_sws_pb_64;

    }

    lastDstY= dstY;



    for (;dstY < dstH; dstY++) {

        const int chrDstY= dstY>>c->chrDstVSubSample;

        uint8_t *dest[4] = {

            dst[0] + dstStride[0] * dstY,

            dst[1] + dstStride[1] * chrDstY,

            dst[2] + dstStride[2] * chrDstY,

            (CONFIG_SWSCALE_ALPHA && alpPixBuf) ? dst[3] + dstStride[3] * dstY : NULL,

        };



        const int firstLumSrcY= FFMAX(1 - vLumFilterSize, vLumFilterPos[dstY]); //First line needed as input

        const int firstLumSrcY2= FFMAX(1 - vLumFilterSize, vLumFilterPos[FFMIN(dstY | ((1<<c->chrDstVSubSample) - 1), dstH-1)]);

        const int firstChrSrcY= FFMAX(1 - vChrFilterSize, vChrFilterPos[chrDstY]); //First line needed as input



        // Last line needed as input

        int lastLumSrcY  = FFMIN(c->srcH,    firstLumSrcY  + vLumFilterSize) - 1;

        int lastLumSrcY2 = FFMIN(c->srcH,    firstLumSrcY2 + vLumFilterSize) - 1;

        int lastChrSrcY  = FFMIN(c->chrSrcH, firstChrSrcY  + vChrFilterSize) - 1;

        int enough_lines;



        //handle holes (FAST_BILINEAR & weird filters)

        if (firstLumSrcY > lastInLumBuf) lastInLumBuf= firstLumSrcY-1;

        if (firstChrSrcY > lastInChrBuf) lastInChrBuf= firstChrSrcY-1;

        assert(firstLumSrcY >= lastInLumBuf - vLumBufSize + 1);

        assert(firstChrSrcY >= lastInChrBuf - vChrBufSize + 1);



        DEBUG_BUFFERS("dstY: %d\n", dstY);

        DEBUG_BUFFERS("\tfirstLumSrcY: %d lastLumSrcY: %d lastInLumBuf: %d\n",

                         firstLumSrcY,    lastLumSrcY,    lastInLumBuf);

        DEBUG_BUFFERS("\tfirstChrSrcY: %d lastChrSrcY: %d lastInChrBuf: %d\n",

                         firstChrSrcY,    lastChrSrcY,    lastInChrBuf);



        // Do we have enough lines in this slice to output the dstY line

        enough_lines = lastLumSrcY2 < srcSliceY + srcSliceH && lastChrSrcY < -((-srcSliceY - srcSliceH)>>c->chrSrcVSubSample);



        if (!enough_lines) {

            lastLumSrcY = srcSliceY + srcSliceH - 1;

            lastChrSrcY = chrSrcSliceY + chrSrcSliceH - 1;

            DEBUG_BUFFERS("buffering slice: lastLumSrcY %d lastChrSrcY %d\n",

                                            lastLumSrcY, lastChrSrcY);

        }



        //Do horizontal scaling

        while(lastInLumBuf < lastLumSrcY) {

            const uint8_t *src1[4] = {

                src[0] + (lastInLumBuf + 1 - srcSliceY) * srcStride[0],

                src[1] + (lastInLumBuf + 1 - srcSliceY) * srcStride[1],

                src[2] + (lastInLumBuf + 1 - srcSliceY) * srcStride[2],

                src[3] + (lastInLumBuf + 1 - srcSliceY) * srcStride[3],

            };

            lumBufIndex++;

            assert(lumBufIndex < 2*vLumBufSize);

            assert(lastInLumBuf + 1 - srcSliceY < srcSliceH);

            assert(lastInLumBuf + 1 - srcSliceY >= 0);

            hyscale(c, lumPixBuf[ lumBufIndex ], dstW, src1, srcW, lumXInc,

                    hLumFilter, hLumFilterPos, hLumFilterSize,

                    formatConvBuffer,

                    pal, 0);

            if (CONFIG_SWSCALE_ALPHA && alpPixBuf)

                hyscale(c, alpPixBuf[ lumBufIndex ], dstW, src1, srcW,

                        lumXInc, hLumFilter, hLumFilterPos, hLumFilterSize,

                        formatConvBuffer,

                        pal, 1);

            lastInLumBuf++;

            DEBUG_BUFFERS("\t\tlumBufIndex %d: lastInLumBuf: %d\n",

                               lumBufIndex,    lastInLumBuf);

        }

        while(lastInChrBuf < lastChrSrcY) {

            const uint8_t *src1[4] = {

                src[0] + (lastInChrBuf + 1 - chrSrcSliceY) * srcStride[0],

                src[1] + (lastInChrBuf + 1 - chrSrcSliceY) * srcStride[1],

                src[2] + (lastInChrBuf + 1 - chrSrcSliceY) * srcStride[2],

                src[3] + (lastInChrBuf + 1 - chrSrcSliceY) * srcStride[3],

            };

            chrBufIndex++;

            assert(chrBufIndex < 2*vChrBufSize);

            assert(lastInChrBuf + 1 - chrSrcSliceY < (chrSrcSliceH));

            assert(lastInChrBuf + 1 - chrSrcSliceY >= 0);

            //FIXME replace parameters through context struct (some at least)



            if (c->needs_hcscale)

                hcscale(c, chrUPixBuf[chrBufIndex], chrVPixBuf[chrBufIndex],

                          chrDstW, src1, chrSrcW, chrXInc,

                          hChrFilter, hChrFilterPos, hChrFilterSize,

                          formatConvBuffer, pal);

            lastInChrBuf++;

            DEBUG_BUFFERS("\t\tchrBufIndex %d: lastInChrBuf: %d\n",

                               chrBufIndex,    lastInChrBuf);

        }

        //wrap buf index around to stay inside the ring buffer

        if (lumBufIndex >= vLumBufSize) lumBufIndex-= vLumBufSize;

        if (chrBufIndex >= vChrBufSize) chrBufIndex-= vChrBufSize;

        if (!enough_lines)

            break; //we can't output a dstY line so let's try with the next slice



#if HAVE_MMX

        updateMMXDitherTables(c, dstY, lumBufIndex, chrBufIndex, lastInLumBuf, lastInChrBuf);

#endif

        if (should_dither) {

            c->chrDither8 = dither_8x8_128[chrDstY & 7];

            c->lumDither8 = dither_8x8_128[dstY & 7];

        }

        if (dstY >= dstH-2) {

            // hmm looks like we can't use MMX here without overwriting this array's tail

            ff_sws_init_output_funcs(c, &yuv2plane1, &yuv2planeX,  &yuv2nv12cX,

                                     &yuv2packed1, &yuv2packed2, &yuv2packedX);

        }



        {

            const int16_t **lumSrcPtr= (const int16_t **) lumPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize;

            const int16_t **chrUSrcPtr= (const int16_t **) chrUPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;

            const int16_t **chrVSrcPtr= (const int16_t **) chrVPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;

            const int16_t **alpSrcPtr= (CONFIG_SWSCALE_ALPHA && alpPixBuf) ? (const int16_t **) alpPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize : NULL;



            if (firstLumSrcY < 0 || firstLumSrcY + vLumFilterSize > c->srcH) {

                const int16_t **tmpY = (const int16_t **) lumPixBuf + 2 * vLumBufSize;

                int neg = -firstLumSrcY, i, end = FFMIN(c->srcH - firstLumSrcY, vLumFilterSize);

                for (i = 0; i < neg;            i++)

                    tmpY[i] = lumSrcPtr[neg];

                for (     ; i < end;            i++)

                    tmpY[i] = lumSrcPtr[i];

                for (     ; i < vLumFilterSize; i++)

                    tmpY[i] = tmpY[i-1];

                lumSrcPtr = tmpY;



                if (alpSrcPtr) {

                    const int16_t **tmpA = (const int16_t **) alpPixBuf + 2 * vLumBufSize;

                    for (i = 0; i < neg;            i++)

                        tmpA[i] = alpSrcPtr[neg];

                    for (     ; i < end;            i++)

                        tmpA[i] = alpSrcPtr[i];

                    for (     ; i < vLumFilterSize; i++)

                        tmpA[i] = tmpA[i - 1];

                    alpSrcPtr = tmpA;

                }

            }

            if (firstChrSrcY < 0 || firstChrSrcY + vChrFilterSize > c->chrSrcH) {

                const int16_t **tmpU = (const int16_t **) chrUPixBuf + 2 * vChrBufSize,

                              **tmpV = (const int16_t **) chrVPixBuf + 2 * vChrBufSize;

                int neg = -firstChrSrcY, i, end = FFMIN(c->chrSrcH - firstChrSrcY, vChrFilterSize);

                for (i = 0; i < neg;            i++) {

                    tmpU[i] = chrUSrcPtr[neg];

                    tmpV[i] = chrVSrcPtr[neg];

                }

                for (     ; i < end;            i++) {

                    tmpU[i] = chrUSrcPtr[i];

                    tmpV[i] = chrVSrcPtr[i];

                }

                for (     ; i < vChrFilterSize; i++) {

                    tmpU[i] = tmpU[i - 1];

                    tmpV[i] = tmpV[i - 1];

                }

                chrUSrcPtr = tmpU;

                chrVSrcPtr = tmpV;

            }



            if (isPlanarYUV(dstFormat) || (isGray(dstFormat) && !isALPHA(dstFormat))) { //YV12 like

                const int chrSkipMask= (1<<c->chrDstVSubSample)-1;



                if (vLumFilterSize == 1) {

                    yuv2plane1(lumSrcPtr[0], dest[0], dstW, c->lumDither8, 0);

                } else {

                    yuv2planeX(vLumFilter + dstY * vLumFilterSize, vLumFilterSize,

                               lumSrcPtr, dest[0], dstW, c->lumDither8, 0);

                }



                if (!((dstY&chrSkipMask) || isGray(dstFormat))) {

                    if (yuv2nv12cX) {

                        yuv2nv12cX(c, vChrFilter + chrDstY * vChrFilterSize, vChrFilterSize, chrUSrcPtr, chrVSrcPtr, dest[1], chrDstW);

                    } else if (vChrFilterSize == 1) {

                        yuv2plane1(chrUSrcPtr[0], dest[1], chrDstW, c->chrDither8, 0);

                        yuv2plane1(chrVSrcPtr[0], dest[2], chrDstW, c->chrDither8, 3);

                    } else {

                        yuv2planeX(vChrFilter + chrDstY * vChrFilterSize, vChrFilterSize,

                                   chrUSrcPtr, dest[1], chrDstW, c->chrDither8, 0);

                        yuv2planeX(vChrFilter + chrDstY * vChrFilterSize, vChrFilterSize,

                                   chrVSrcPtr, dest[2], chrDstW, c->chrDither8, 3);

                    }

                }



                if (CONFIG_SWSCALE_ALPHA && alpPixBuf){

                    if (vLumFilterSize == 1) {

                        yuv2plane1(alpSrcPtr[0], dest[3], dstW, c->lumDither8, 0);

                    } else {

                        yuv2planeX(vLumFilter + dstY * vLumFilterSize, vLumFilterSize,

                                   alpSrcPtr, dest[3], dstW, c->lumDither8, 0);

                    }

                }

            } else {

                assert(lumSrcPtr  + vLumFilterSize - 1 < lumPixBuf  + vLumBufSize*2);

                assert(chrUSrcPtr + vChrFilterSize - 1 < chrUPixBuf + vChrBufSize*2);

                if (c->yuv2packed1 && vLumFilterSize == 1 && vChrFilterSize <= 2) { //unscaled RGB

                    int chrAlpha = vChrFilterSize == 1 ? 0 : vChrFilter[2 * dstY + 1];

                    yuv2packed1(c, *lumSrcPtr, chrUSrcPtr, chrVSrcPtr,

                                alpPixBuf ? *alpSrcPtr : NULL,

                                dest[0], dstW, chrAlpha, dstY);

                } else if (c->yuv2packed2 && vLumFilterSize == 2 && vChrFilterSize == 2) { //bilinear upscale RGB

                    int lumAlpha = vLumFilter[2 * dstY + 1];

                    int chrAlpha = vChrFilter[2 * dstY + 1];

                    lumMmxFilter[2] =

                    lumMmxFilter[3] = vLumFilter[2 * dstY   ] * 0x10001;

                    chrMmxFilter[2] =

                    chrMmxFilter[3] = vChrFilter[2 * chrDstY] * 0x10001;

                    yuv2packed2(c, lumSrcPtr, chrUSrcPtr, chrVSrcPtr,

                                alpPixBuf ? alpSrcPtr : NULL,

                                dest[0], dstW, lumAlpha, chrAlpha, dstY);

                } else { //general RGB

                    yuv2packedX(c, vLumFilter + dstY * vLumFilterSize,

                                lumSrcPtr, vLumFilterSize,

                                vChrFilter + dstY * vChrFilterSize,

                                chrUSrcPtr, chrVSrcPtr, vChrFilterSize,

                                alpSrcPtr, dest[0], dstW, dstY);

                }

            }

        }

    }



    if (isPlanar(dstFormat) && isALPHA(dstFormat) && !alpPixBuf)

        fillPlane(dst[3], dstStride[3], dstW, dstY-lastDstY, lastDstY, 255);



#if HAVE_MMX2

    if (av_get_cpu_flags() & AV_CPU_FLAG_MMX2)

        __asm__ volatile("sfence":::"memory");

#endif

    emms_c();



    /* store changed local vars back in the context */

    c->dstY= dstY;

    c->lumBufIndex= lumBufIndex;

    c->chrBufIndex= chrBufIndex;

    c->lastInLumBuf= lastInLumBuf;

    c->lastInChrBuf= lastInChrBuf;



    return dstY - lastDstY;

}
