static int RENAME(swScale)(SwsContext *c, uint8_t* src[], int srcStride[], int srcSliceY,

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

    const enum PixelFormat srcFormat= c->srcFormat;

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

    int32_t *alpMmxFilter= c->alpMmxFilter;

    const int vLumFilterSize= c->vLumFilterSize;

    const int vChrFilterSize= c->vChrFilterSize;

    const int hLumFilterSize= c->hLumFilterSize;

    const int hChrFilterSize= c->hChrFilterSize;

    int16_t **lumPixBuf= c->lumPixBuf;

    int16_t **chrPixBuf= c->chrPixBuf;

    int16_t **alpPixBuf= c->alpPixBuf;

    const int vLumBufSize= c->vLumBufSize;

    const int vChrBufSize= c->vChrBufSize;

    uint8_t *formatConvBuffer= c->formatConvBuffer;

    const int chrSrcSliceY= srcSliceY >> c->chrSrcVSubSample;

    const int chrSrcSliceH= -((-srcSliceH) >> c->chrSrcVSubSample);

    int lastDstY;

    uint32_t *pal=c->pal_yuv;



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

        lumBufIndex=0;

        chrBufIndex=0;

        dstY=0;

        lastInLumBuf= -1;

        lastInChrBuf= -1;

    }



    lastDstY= dstY;



    for (;dstY < dstH; dstY++) {

        unsigned char *dest =dst[0]+dstStride[0]*dstY;

        const int chrDstY= dstY>>c->chrDstVSubSample;

        unsigned char *uDest=dst[1]+dstStride[1]*chrDstY;

        unsigned char *vDest=dst[2]+dstStride[2]*chrDstY;

        unsigned char *aDest=(CONFIG_SWSCALE_ALPHA && alpPixBuf) ? dst[3]+dstStride[3]*dstY : NULL;



        const int firstLumSrcY= vLumFilterPos[dstY]; //First line needed as input

        const int firstChrSrcY= vChrFilterPos[chrDstY]; //First line needed as input

        int lastLumSrcY= firstLumSrcY + vLumFilterSize -1; // Last line needed as input

        int lastChrSrcY= firstChrSrcY + vChrFilterSize -1; // Last line needed as input

        int enough_lines;



        //handle holes (FAST_BILINEAR & weird filters)

        if (firstLumSrcY > lastInLumBuf) lastInLumBuf= firstLumSrcY-1;

        if (firstChrSrcY > lastInChrBuf) lastInChrBuf= firstChrSrcY-1;

        assert(firstLumSrcY >= lastInLumBuf - vLumBufSize + 1);

        assert(firstChrSrcY >= lastInChrBuf - vChrBufSize + 1);



        // Do we have enough lines in this slice to output the dstY line

        enough_lines = lastLumSrcY < srcSliceY + srcSliceH && lastChrSrcY < -((-srcSliceY - srcSliceH)>>c->chrSrcVSubSample);

        if (!enough_lines) {

            lastLumSrcY = srcSliceY + srcSliceH - 1;

            lastChrSrcY = chrSrcSliceY + chrSrcSliceH - 1;

        }



        DEBUG_BUFFERS("dstY: %d\n", dstY);

        DEBUG_BUFFERS("\tfirstLumSrcY: %d lastLumSrcY: %d lastInLumBuf: %d\n",

                         firstLumSrcY,    lastLumSrcY,    lastInLumBuf);

        DEBUG_BUFFERS("\tfirstChrSrcY: %d lastChrSrcY: %d lastInChrBuf: %d\n",

                         firstChrSrcY,    lastChrSrcY,    lastInChrBuf);



        //Do horizontal scaling

        while(lastInLumBuf < lastLumSrcY) {

            uint8_t *src1= src[0]+(lastInLumBuf + 1 - srcSliceY)*srcStride[0];

            uint8_t *src2= src[3]+(lastInLumBuf + 1 - srcSliceY)*srcStride[3];

            lumBufIndex++;

            DEBUG_BUFFERS("\t\tlumBufIndex %d: lastInLumBuf: %d\n",

                               lumBufIndex,    lastInLumBuf);

            assert(lumBufIndex < 2*vLumBufSize);

            assert(lastInLumBuf + 1 - srcSliceY < srcSliceH);

            assert(lastInLumBuf + 1 - srcSliceY >= 0);

            RENAME(hyscale)(c, lumPixBuf[ lumBufIndex ], dstW, src1, srcW, lumXInc,

                            flags, hLumFilter, hLumFilterPos, hLumFilterSize,

                            c->srcFormat, formatConvBuffer,

                            pal, 0);

            if (CONFIG_SWSCALE_ALPHA && alpPixBuf)

                RENAME(hyscale)(c, alpPixBuf[ lumBufIndex ], dstW, src2, srcW, lumXInc,

                                flags, hLumFilter, hLumFilterPos, hLumFilterSize,

                                c->srcFormat, formatConvBuffer,

                                pal, 1);

            lastInLumBuf++;

        }

        while(lastInChrBuf < lastChrSrcY) {

            uint8_t *src1= src[1]+(lastInChrBuf + 1 - chrSrcSliceY)*srcStride[1];

            uint8_t *src2= src[2]+(lastInChrBuf + 1 - chrSrcSliceY)*srcStride[2];

            chrBufIndex++;

            DEBUG_BUFFERS("\t\tchrBufIndex %d: lastInChrBuf: %d\n",

                               chrBufIndex,    lastInChrBuf);

            assert(chrBufIndex < 2*vChrBufSize);

            assert(lastInChrBuf + 1 - chrSrcSliceY < (chrSrcSliceH));

            assert(lastInChrBuf + 1 - chrSrcSliceY >= 0);

            //FIXME replace parameters through context struct (some at least)



            if (!(isGray(srcFormat) || isGray(dstFormat)))

                RENAME(hcscale)(c, chrPixBuf[ chrBufIndex ], chrDstW, src1, src2, chrSrcW, chrXInc,

                                flags, hChrFilter, hChrFilterPos, hChrFilterSize,

                                c->srcFormat, formatConvBuffer,

                                pal);

            lastInChrBuf++;

        }

        //wrap buf index around to stay inside the ring buffer

        if (lumBufIndex >= vLumBufSize) lumBufIndex-= vLumBufSize;

        if (chrBufIndex >= vChrBufSize) chrBufIndex-= vChrBufSize;

        if (!enough_lines)

            break; //we can't output a dstY line so let's try with the next slice



#if COMPILE_TEMPLATE_MMX

        c->blueDither= ff_dither8[dstY&1];

        if (c->dstFormat == PIX_FMT_RGB555 || c->dstFormat == PIX_FMT_BGR555)

            c->greenDither= ff_dither8[dstY&1];

        else

            c->greenDither= ff_dither4[dstY&1];

        c->redDither= ff_dither8[(dstY+1)&1];

#endif

        if (dstY < dstH-2) {

            const int16_t **lumSrcPtr= (const int16_t **) lumPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize;

            const int16_t **chrSrcPtr= (const int16_t **) chrPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;

            const int16_t **alpSrcPtr= (CONFIG_SWSCALE_ALPHA && alpPixBuf) ? (const int16_t **) alpPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize : NULL;

#if COMPILE_TEMPLATE_MMX

            int i;

            if (flags & SWS_ACCURATE_RND) {

                int s= APCK_SIZE / 8;

                for (i=0; i<vLumFilterSize; i+=2) {

                    *(void**)&lumMmxFilter[s*i              ]= lumSrcPtr[i  ];

                    *(void**)&lumMmxFilter[s*i+APCK_PTR2/4  ]= lumSrcPtr[i+(vLumFilterSize>1)];

                              lumMmxFilter[s*i+APCK_COEF/4  ]=

                              lumMmxFilter[s*i+APCK_COEF/4+1]= vLumFilter[dstY*vLumFilterSize + i    ]

                        + (vLumFilterSize>1 ? vLumFilter[dstY*vLumFilterSize + i + 1]<<16 : 0);

                    if (CONFIG_SWSCALE_ALPHA && alpPixBuf) {

                        *(void**)&alpMmxFilter[s*i              ]= alpSrcPtr[i  ];

                        *(void**)&alpMmxFilter[s*i+APCK_PTR2/4  ]= alpSrcPtr[i+(vLumFilterSize>1)];

                                  alpMmxFilter[s*i+APCK_COEF/4  ]=

                                  alpMmxFilter[s*i+APCK_COEF/4+1]= lumMmxFilter[s*i+APCK_COEF/4  ];

                    }

                }

                for (i=0; i<vChrFilterSize; i+=2) {

                    *(void**)&chrMmxFilter[s*i              ]= chrSrcPtr[i  ];

                    *(void**)&chrMmxFilter[s*i+APCK_PTR2/4  ]= chrSrcPtr[i+(vChrFilterSize>1)];

                              chrMmxFilter[s*i+APCK_COEF/4  ]=

                              chrMmxFilter[s*i+APCK_COEF/4+1]= vChrFilter[chrDstY*vChrFilterSize + i    ]

                        + (vChrFilterSize>1 ? vChrFilter[chrDstY*vChrFilterSize + i + 1]<<16 : 0);

                }

            } else {

                for (i=0; i<vLumFilterSize; i++) {

                    lumMmxFilter[4*i+0]= (int32_t)lumSrcPtr[i];

                    lumMmxFilter[4*i+1]= (uint64_t)lumSrcPtr[i] >> 32;

                    lumMmxFilter[4*i+2]=

                    lumMmxFilter[4*i+3]=

                        ((uint16_t)vLumFilter[dstY*vLumFilterSize + i])*0x10001;

                    if (CONFIG_SWSCALE_ALPHA && alpPixBuf) {

                        alpMmxFilter[4*i+0]= (int32_t)alpSrcPtr[i];

                        alpMmxFilter[4*i+1]= (uint64_t)alpSrcPtr[i] >> 32;

                        alpMmxFilter[4*i+2]=

                        alpMmxFilter[4*i+3]= lumMmxFilter[4*i+2];

                    }

                }

                for (i=0; i<vChrFilterSize; i++) {

                    chrMmxFilter[4*i+0]= (int32_t)chrSrcPtr[i];

                    chrMmxFilter[4*i+1]= (uint64_t)chrSrcPtr[i] >> 32;

                    chrMmxFilter[4*i+2]=

                    chrMmxFilter[4*i+3]=

                        ((uint16_t)vChrFilter[chrDstY*vChrFilterSize + i])*0x10001;

                }

            }

#endif

            if (dstFormat == PIX_FMT_NV12 || dstFormat == PIX_FMT_NV21) {

                const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

                if (dstY&chrSkipMask) uDest= NULL; //FIXME split functions in lumi / chromi

                c->yuv2nv12X(c,

                             vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

                             vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                             dest, uDest, dstW, chrDstW, dstFormat);

            } else if (isPlanarYUV(dstFormat) || dstFormat==PIX_FMT_GRAY8) { //YV12 like

                const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

                if ((dstY&chrSkipMask) || isGray(dstFormat)) uDest=vDest= NULL; //FIXME split functions in lumi / chromi

                if (is16BPS(dstFormat)) {

                    yuv2yuvX16inC(

                                  vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

                                  vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                  alpSrcPtr, (uint16_t *) dest, (uint16_t *) uDest, (uint16_t *) vDest, (uint16_t *) aDest, dstW, chrDstW,

                                  dstFormat);

                } else if (vLumFilterSize == 1 && vChrFilterSize == 1) { // unscaled YV12

                    int16_t *lumBuf = lumSrcPtr[0];

                    int16_t *chrBuf= chrSrcPtr[0];

                    int16_t *alpBuf= (CONFIG_SWSCALE_ALPHA && alpPixBuf) ? alpSrcPtr[0] : NULL;

                    c->yuv2yuv1(c, lumBuf, chrBuf, alpBuf, dest, uDest, vDest, aDest, dstW, chrDstW);

                } else { //General YV12

                    c->yuv2yuvX(c,

                                vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

                                vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                alpSrcPtr, dest, uDest, vDest, aDest, dstW, chrDstW);

                }

            } else {

                assert(lumSrcPtr + vLumFilterSize - 1 < lumPixBuf + vLumBufSize*2);

                assert(chrSrcPtr + vChrFilterSize - 1 < chrPixBuf + vChrBufSize*2);

                if (vLumFilterSize == 1 && vChrFilterSize == 2) { //unscaled RGB

                    int chrAlpha= vChrFilter[2*dstY+1];

                    if(flags & SWS_FULL_CHR_H_INT) {

                        yuv2rgbXinC_full(c, //FIXME write a packed1_full function

                                         vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

                                         vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                         alpSrcPtr, dest, dstW, dstY);

                    } else {

                        c->yuv2packed1(c, *lumSrcPtr, *chrSrcPtr, *(chrSrcPtr+1),

                                       alpPixBuf ? *alpSrcPtr : NULL,

                                       dest, dstW, chrAlpha, dstFormat, flags, dstY);

                    }

                } else if (vLumFilterSize == 2 && vChrFilterSize == 2) { //bilinear upscale RGB

                    int lumAlpha= vLumFilter[2*dstY+1];

                    int chrAlpha= vChrFilter[2*dstY+1];

                    lumMmxFilter[2]=

                    lumMmxFilter[3]= vLumFilter[2*dstY   ]*0x10001;

                    chrMmxFilter[2]=

                    chrMmxFilter[3]= vChrFilter[2*chrDstY]*0x10001;

                    if(flags & SWS_FULL_CHR_H_INT) {

                        yuv2rgbXinC_full(c, //FIXME write a packed2_full function

                                         vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

                                         vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                         alpSrcPtr, dest, dstW, dstY);

                    } else {

                        c->yuv2packed2(c, *lumSrcPtr, *(lumSrcPtr+1), *chrSrcPtr, *(chrSrcPtr+1),

                                       alpPixBuf ? *alpSrcPtr : NULL, alpPixBuf ? *(alpSrcPtr+1) : NULL,

                                       dest, dstW, lumAlpha, chrAlpha, dstY);

                    }

                } else { //general RGB

                    if(flags & SWS_FULL_CHR_H_INT) {

                        yuv2rgbXinC_full(c,

                                         vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

                                         vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                         alpSrcPtr, dest, dstW, dstY);

                    } else {

                        c->yuv2packedX(c,

                                       vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

                                       vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                       alpSrcPtr, dest, dstW, dstY);

                    }

                }

            }

        } else { // hmm looks like we can't use MMX here without overwriting this array's tail

            const int16_t **lumSrcPtr= (const int16_t **)lumPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize;

            const int16_t **chrSrcPtr= (const int16_t **)chrPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;

            const int16_t **alpSrcPtr= (CONFIG_SWSCALE_ALPHA && alpPixBuf) ? (const int16_t **)alpPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize : NULL;

            if (dstFormat == PIX_FMT_NV12 || dstFormat == PIX_FMT_NV21) {

                const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

                if (dstY&chrSkipMask) uDest= NULL; //FIXME split functions in lumi / chromi

                yuv2nv12XinC(

                             vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

                             vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                             dest, uDest, dstW, chrDstW, dstFormat);

            } else if (isPlanarYUV(dstFormat) || dstFormat==PIX_FMT_GRAY8) { //YV12

                const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

                if ((dstY&chrSkipMask) || isGray(dstFormat)) uDest=vDest= NULL; //FIXME split functions in lumi / chromi

                if (is16BPS(dstFormat)) {

                    yuv2yuvX16inC(

                                  vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

                                  vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                  alpSrcPtr, (uint16_t *) dest, (uint16_t *) uDest, (uint16_t *) vDest, (uint16_t *) aDest, dstW, chrDstW,

                                  dstFormat);

                } else {

                    yuv2yuvXinC(

                                vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

                                vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                alpSrcPtr, dest, uDest, vDest, aDest, dstW, chrDstW);

                }

            } else {

                assert(lumSrcPtr + vLumFilterSize - 1 < lumPixBuf + vLumBufSize*2);

                assert(chrSrcPtr + vChrFilterSize - 1 < chrPixBuf + vChrBufSize*2);

                if(flags & SWS_FULL_CHR_H_INT) {

                    yuv2rgbXinC_full(c,

                                     vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

                                     vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                     alpSrcPtr, dest, dstW, dstY);

                } else {

                    yuv2packedXinC(c,

                                   vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

                                   vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

                                   alpSrcPtr, dest, dstW, dstY);

                }

            }

        }

    }



    if ((dstFormat == PIX_FMT_YUVA420P) && !alpPixBuf)

        fillPlane(dst[3], dstStride[3], dstW, dstY-lastDstY, lastDstY, 255);



#if COMPILE_TEMPLATE_MMX

    if (flags & SWS_CPU_CAPS_MMX2 )  __asm__ volatile("sfence":::"memory");

    /* On K6 femms is faster than emms. On K7 femms is directly mapped to emms. */

    if (flags & SWS_CPU_CAPS_3DNOW)  __asm__ volatile("femms" :::"memory");

    else                             __asm__ volatile("emms"  :::"memory");

#endif

    /* store changed local vars back in the context */

    c->dstY= dstY;

    c->lumBufIndex= lumBufIndex;

    c->chrBufIndex= chrBufIndex;

    c->lastInLumBuf= lastInLumBuf;

    c->lastInChrBuf= lastInChrBuf;



    return dstY - lastDstY;

}
