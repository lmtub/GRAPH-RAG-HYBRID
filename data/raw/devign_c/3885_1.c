av_cold int sws_init_context(SwsContext *c, SwsFilter *srcFilter,

                             SwsFilter *dstFilter)

{

    int i;

    int usesVFilter, usesHFilter;

    int unscaled;

    SwsFilter dummyFilter = { NULL, NULL, NULL, NULL };

    int srcW              = c->srcW;

    int srcH              = c->srcH;

    int dstW              = c->dstW;

    int dstH              = c->dstH;

    int dst_stride        = FFALIGN(dstW * sizeof(int16_t) + 16, 16);

    int dst_stride_px     = dst_stride >> 1;

    int flags, cpu_flags;

    enum AVPixelFormat srcFormat = c->srcFormat;

    enum AVPixelFormat dstFormat = c->dstFormat;

    const AVPixFmtDescriptor *desc_src = av_pix_fmt_desc_get(srcFormat);

    const AVPixFmtDescriptor *desc_dst = av_pix_fmt_desc_get(dstFormat);



    cpu_flags = av_get_cpu_flags();

    flags     = c->flags;

    emms_c();

    if (!rgb15to16)

        ff_rgb2rgb_init();



    unscaled = (srcW == dstW && srcH == dstH);



    if (!(unscaled && sws_isSupportedEndiannessConversion(srcFormat) &&

          av_pix_fmt_swap_endianness(srcFormat) == dstFormat)) {

    if (!sws_isSupportedInput(srcFormat)) {

        av_log(c, AV_LOG_ERROR, "%s is not supported as input pixel format\n",

               sws_format_name(srcFormat));

        return AVERROR(EINVAL);

    }

    if (!sws_isSupportedOutput(dstFormat)) {

        av_log(c, AV_LOG_ERROR, "%s is not supported as output pixel format\n",

               sws_format_name(dstFormat));

        return AVERROR(EINVAL);

    }

    }



    i = flags & (SWS_POINT         |

                 SWS_AREA          |

                 SWS_BILINEAR      |

                 SWS_FAST_BILINEAR |

                 SWS_BICUBIC       |

                 SWS_X             |

                 SWS_GAUSS         |

                 SWS_LANCZOS       |

                 SWS_SINC          |

                 SWS_SPLINE        |

                 SWS_BICUBLIN);



    /* provide a default scaler if not set by caller */

    if (!i) {

        if (dstW < srcW && dstH < srcH)

            flags |= SWS_GAUSS;

        else if (dstW > srcW && dstH > srcH)

            flags |= SWS_SINC;

        else

            flags |= SWS_LANCZOS;

        c->flags = flags;

    } else if (i & (i - 1)) {

        av_log(c, AV_LOG_ERROR,

               "Exactly one scaler algorithm must be chosen\n");

        return AVERROR(EINVAL);

    }

    /* sanity check */

    if (srcW < 4 || srcH < 1 || dstW < 8 || dstH < 1) {

        /* FIXME check if these are enough and try to lower them after

         * fixing the relevant parts of the code */

        av_log(c, AV_LOG_ERROR, "%dx%d -> %dx%d is invalid scaling dimension\n",

               srcW, srcH, dstW, dstH);

        return AVERROR(EINVAL);

    }



    if (!dstFilter)

        dstFilter = &dummyFilter;

    if (!srcFilter)

        srcFilter = &dummyFilter;



    c->lumXInc      = (((int64_t)srcW << 16) + (dstW >> 1)) / dstW;

    c->lumYInc      = (((int64_t)srcH << 16) + (dstH >> 1)) / dstH;

    c->dstFormatBpp = av_get_bits_per_pixel(desc_dst);

    c->srcFormatBpp = av_get_bits_per_pixel(desc_src);

    c->vRounder     = 4 * 0x0001000100010001ULL;



    usesVFilter = (srcFilter->lumV && srcFilter->lumV->length > 1) ||

                  (srcFilter->chrV && srcFilter->chrV->length > 1) ||

                  (dstFilter->lumV && dstFilter->lumV->length > 1) ||

                  (dstFilter->chrV && dstFilter->chrV->length > 1);

    usesHFilter = (srcFilter->lumH && srcFilter->lumH->length > 1) ||

                  (srcFilter->chrH && srcFilter->chrH->length > 1) ||

                  (dstFilter->lumH && dstFilter->lumH->length > 1) ||

                  (dstFilter->chrH && dstFilter->chrH->length > 1);



    getSubSampleFactors(&c->chrSrcHSubSample, &c->chrSrcVSubSample, srcFormat);

    getSubSampleFactors(&c->chrDstHSubSample, &c->chrDstVSubSample, dstFormat);



    if (isPlanarRGB(dstFormat)) {

        if (!(flags & SWS_FULL_CHR_H_INT)) {

            av_log(c, AV_LOG_DEBUG,

                   "%s output is not supported with half chroma resolution, switching to full\n",

                   av_get_pix_fmt_name(dstFormat));

            flags   |= SWS_FULL_CHR_H_INT;

            c->flags = flags;

        }

    }



    /* reuse chroma for 2 pixels RGB/BGR unless user wants full

     * chroma interpolation */

    if (flags & SWS_FULL_CHR_H_INT &&

        isAnyRGB(dstFormat)        &&

        !isPlanarRGB(dstFormat)    &&

        dstFormat != AV_PIX_FMT_RGBA  &&

        dstFormat != AV_PIX_FMT_ARGB  &&

        dstFormat != AV_PIX_FMT_BGRA  &&

        dstFormat != AV_PIX_FMT_ABGR  &&

        dstFormat != AV_PIX_FMT_RGB24 &&

        dstFormat != AV_PIX_FMT_BGR24) {

        av_log(c, AV_LOG_ERROR,

               "full chroma interpolation for destination format '%s' not yet implemented\n",

               sws_format_name(dstFormat));

        flags   &= ~SWS_FULL_CHR_H_INT;

        c->flags = flags;

    }

    if (isAnyRGB(dstFormat) && !(flags & SWS_FULL_CHR_H_INT))

        c->chrDstHSubSample = 1;



    // drop some chroma lines if the user wants it

    c->vChrDrop          = (flags & SWS_SRC_V_CHR_DROP_MASK) >>

                           SWS_SRC_V_CHR_DROP_SHIFT;

    c->chrSrcVSubSample += c->vChrDrop;



    /* drop every other pixel for chroma calculation unless user

     * wants full chroma */

    if (isAnyRGB(srcFormat) && !(flags & SWS_FULL_CHR_H_INP)   &&

        srcFormat != AV_PIX_FMT_RGB8 && srcFormat != AV_PIX_FMT_BGR8 &&

        srcFormat != AV_PIX_FMT_RGB4 && srcFormat != AV_PIX_FMT_BGR4 &&

        srcFormat != AV_PIX_FMT_RGB4_BYTE && srcFormat != AV_PIX_FMT_BGR4_BYTE &&

        srcFormat != AV_PIX_FMT_GBRP9BE   && srcFormat != AV_PIX_FMT_GBRP9LE  &&

        srcFormat != AV_PIX_FMT_GBRP10BE  && srcFormat != AV_PIX_FMT_GBRP10LE &&

        srcFormat != AV_PIX_FMT_GBRAP10BE && srcFormat != AV_PIX_FMT_GBRAP10LE &&

        srcFormat != AV_PIX_FMT_GBRP12BE  && srcFormat != AV_PIX_FMT_GBRP12LE &&

        srcFormat != AV_PIX_FMT_GBRP16BE  && srcFormat != AV_PIX_FMT_GBRP16LE &&

        ((dstW >> c->chrDstHSubSample) <= (srcW >> 1) ||

         (flags & SWS_FAST_BILINEAR)))

        c->chrSrcHSubSample = 1;



    // Note the AV_CEIL_RSHIFT is so that we always round toward +inf.

    c->chrSrcW = AV_CEIL_RSHIFT(srcW, c->chrSrcHSubSample);

    c->chrSrcH = AV_CEIL_RSHIFT(srcH, c->chrSrcVSubSample);

    c->chrDstW = AV_CEIL_RSHIFT(dstW, c->chrDstHSubSample);

    c->chrDstH = AV_CEIL_RSHIFT(dstH, c->chrDstVSubSample);



    /* unscaled special cases */

    if (unscaled && !usesHFilter && !usesVFilter &&

        (c->srcRange == c->dstRange || isAnyRGB(dstFormat))) {

        ff_get_unscaled_swscale(c);



        if (c->swscale) {

            if (flags & SWS_PRINT_INFO)

                av_log(c, AV_LOG_INFO,

                       "using unscaled %s -> %s special converter\n",

                       sws_format_name(srcFormat), sws_format_name(dstFormat));

            return 0;

        }

    }



    c->srcBpc = desc_src->comp[0].depth;

    if (c->srcBpc < 8)

        c->srcBpc = 8;

    c->dstBpc = desc_dst->comp[0].depth;

    if (c->dstBpc < 8)

        c->dstBpc = 8;

    if (c->dstBpc == 16)

        dst_stride <<= 1;

    FF_ALLOC_OR_GOTO(c, c->formatConvBuffer,

                     (FFALIGN(srcW, 16) * 2 * FFALIGN(c->srcBpc, 8) >> 3) + 16,

                     fail);

    if (INLINE_MMXEXT(cpu_flags) && c->srcBpc == 8 && c->dstBpc <= 12) {

        c->canMMXEXTBeUsed = (dstW >= srcW && (dstW & 31) == 0 &&

                              (srcW & 15) == 0) ? 1 : 0;

        if (!c->canMMXEXTBeUsed && dstW >= srcW && (srcW & 15) == 0

            && (flags & SWS_FAST_BILINEAR)) {

            if (flags & SWS_PRINT_INFO)

                av_log(c, AV_LOG_INFO,

                       "output width is not a multiple of 32 -> no MMXEXT scaler\n");

        }

        if (usesHFilter)

            c->canMMXEXTBeUsed = 0;

    } else

        c->canMMXEXTBeUsed = 0;



    c->chrXInc = (((int64_t)c->chrSrcW << 16) + (c->chrDstW >> 1)) / c->chrDstW;

    c->chrYInc = (((int64_t)c->chrSrcH << 16) + (c->chrDstH >> 1)) / c->chrDstH;



    /* Match pixel 0 of the src to pixel 0 of dst and match pixel n-2 of src

     * to pixel n-2 of dst, but only for the FAST_BILINEAR mode otherwise do

     * correct scaling.

     * n-2 is the last chrominance sample available.

     * This is not perfect, but no one should notice the difference, the more

     * correct variant would be like the vertical one, but that would require

     * some special code for the first and last pixel */

    if (flags & SWS_FAST_BILINEAR) {

        if (c->canMMXEXTBeUsed) {

            c->lumXInc += 20;

            c->chrXInc += 20;

        }

        // we don't use the x86 asm scaler if MMX is available

        else if (INLINE_MMX(cpu_flags)) {

            c->lumXInc = ((int64_t)(srcW       - 2) << 16) / (dstW       - 2) - 20;

            c->chrXInc = ((int64_t)(c->chrSrcW - 2) << 16) / (c->chrDstW - 2) - 20;

        }

    }



#define USE_MMAP (HAVE_MMAP && HAVE_MPROTECT && defined MAP_ANONYMOUS)



    /* precalculate horizontal scaler filter coefficients */

    {

#if HAVE_MMXEXT_INLINE

// can't downscale !!!

        if (c->canMMXEXTBeUsed && (flags & SWS_FAST_BILINEAR)) {

            c->lumMmxextFilterCodeSize = init_hscaler_mmxext(dstW, c->lumXInc, NULL,

                                                             NULL, NULL, 8);

            c->chrMmxextFilterCodeSize = init_hscaler_mmxext(c->chrDstW, c->chrXInc,

                                                             NULL, NULL, NULL, 4);



#if USE_MMAP

            c->lumMmxextFilterCode = mmap(NULL, c->lumMmxextFilterCodeSize,

                                          PROT_READ | PROT_WRITE,

                                          MAP_PRIVATE | MAP_ANONYMOUS,

                                          -1, 0);

            c->chrMmxextFilterCode = mmap(NULL, c->chrMmxextFilterCodeSize,

                                          PROT_READ | PROT_WRITE,

                                          MAP_PRIVATE | MAP_ANONYMOUS,

                                          -1, 0);

#elif HAVE_VIRTUALALLOC

            c->lumMmxextFilterCode = VirtualAlloc(NULL,

                                                  c->lumMmxextFilterCodeSize,

                                                  MEM_COMMIT,

                                                  PAGE_EXECUTE_READWRITE);

            c->chrMmxextFilterCode = VirtualAlloc(NULL,

                                                  c->chrMmxextFilterCodeSize,

                                                  MEM_COMMIT,

                                                  PAGE_EXECUTE_READWRITE);

#else

            c->lumMmxextFilterCode = av_malloc(c->lumMmxextFilterCodeSize);

            c->chrMmxextFilterCode = av_malloc(c->chrMmxextFilterCodeSize);

#endif



            if (!c->lumMmxextFilterCode || !c->chrMmxextFilterCode)

                return AVERROR(ENOMEM);

            FF_ALLOCZ_OR_GOTO(c, c->hLumFilter,    (dstW           / 8 + 8) * sizeof(int16_t), fail);

            FF_ALLOCZ_OR_GOTO(c, c->hChrFilter,    (c->chrDstW     / 4 + 8) * sizeof(int16_t), fail);

            FF_ALLOCZ_OR_GOTO(c, c->hLumFilterPos, (dstW       / 2 / 8 + 8) * sizeof(int32_t), fail);

            FF_ALLOCZ_OR_GOTO(c, c->hChrFilterPos, (c->chrDstW / 2 / 4 + 8) * sizeof(int32_t), fail);



            init_hscaler_mmxext(dstW, c->lumXInc, c->lumMmxextFilterCode,

                                c->hLumFilter, c->hLumFilterPos, 8);

            init_hscaler_mmxext(c->chrDstW, c->chrXInc, c->chrMmxextFilterCode,

                                c->hChrFilter, c->hChrFilterPos, 4);



#if USE_MMAP

            mprotect(c->lumMmxextFilterCode, c->lumMmxextFilterCodeSize, PROT_EXEC | PROT_READ);

            mprotect(c->chrMmxextFilterCode, c->chrMmxextFilterCodeSize, PROT_EXEC | PROT_READ);

#endif

        } else

#endif /* HAVE_MMXEXT_INLINE */

        {

            const int filterAlign = X86_MMX(cpu_flags)     ? 4 :

                                    PPC_ALTIVEC(cpu_flags) ? 8 : 1;



            if (initFilter(&c->hLumFilter, &c->hLumFilterPos,

                           &c->hLumFilterSize, c->lumXInc,

                           srcW, dstW, filterAlign, 1 << 14,

                           (flags & SWS_BICUBLIN) ? (flags | SWS_BICUBIC) : flags,

                           cpu_flags, srcFilter->lumH, dstFilter->lumH,

                           c->param, 1) < 0)

                goto fail;

            if (initFilter(&c->hChrFilter, &c->hChrFilterPos,

                           &c->hChrFilterSize, c->chrXInc,

                           c->chrSrcW, c->chrDstW, filterAlign, 1 << 14,

                           (flags & SWS_BICUBLIN) ? (flags | SWS_BILINEAR) : flags,

                           cpu_flags, srcFilter->chrH, dstFilter->chrH,

                           c->param, 1) < 0)

                goto fail;

        }

    } // initialize horizontal stuff



    /* precalculate vertical scaler filter coefficients */

    {

        const int filterAlign = X86_MMX(cpu_flags)     ? 2 :

                                PPC_ALTIVEC(cpu_flags) ? 8 : 1;



        if (initFilter(&c->vLumFilter, &c->vLumFilterPos, &c->vLumFilterSize,

                       c->lumYInc, srcH, dstH, filterAlign, (1 << 12),

                       (flags & SWS_BICUBLIN) ? (flags | SWS_BICUBIC) : flags,

                       cpu_flags, srcFilter->lumV, dstFilter->lumV,

                       c->param, 0) < 0)

            goto fail;

        if (initFilter(&c->vChrFilter, &c->vChrFilterPos, &c->vChrFilterSize,

                       c->chrYInc, c->chrSrcH, c->chrDstH,

                       filterAlign, (1 << 12),

                       (flags & SWS_BICUBLIN) ? (flags | SWS_BILINEAR) : flags,

                       cpu_flags, srcFilter->chrV, dstFilter->chrV,

                       c->param, 0) < 0)

            goto fail;



#if HAVE_ALTIVEC

        FF_ALLOC_OR_GOTO(c, c->vYCoeffsBank, sizeof(vector signed short) * c->vLumFilterSize * c->dstH,    fail);

        FF_ALLOC_OR_GOTO(c, c->vCCoeffsBank, sizeof(vector signed short) * c->vChrFilterSize * c->chrDstH, fail);



        for (i = 0; i < c->vLumFilterSize * c->dstH; i++) {

            int j;

            short *p = (short *)&c->vYCoeffsBank[i];

            for (j = 0; j < 8; j++)

                p[j] = c->vLumFilter[i];

        }



        for (i = 0; i < c->vChrFilterSize * c->chrDstH; i++) {

            int j;

            short *p = (short *)&c->vCCoeffsBank[i];

            for (j = 0; j < 8; j++)

                p[j] = c->vChrFilter[i];

        }

#endif

    }



    // calculate buffer sizes so that they won't run out while handling these damn slices

    c->vLumBufSize = c->vLumFilterSize;

    c->vChrBufSize = c->vChrFilterSize;

    for (i = 0; i < dstH; i++) {

        int chrI      = (int64_t)i * c->chrDstH / dstH;

        int nextSlice = FFMAX(c->vLumFilterPos[i] + c->vLumFilterSize - 1,

                              ((c->vChrFilterPos[chrI] + c->vChrFilterSize - 1)

                               << c->chrSrcVSubSample));



        nextSlice >>= c->chrSrcVSubSample;

        nextSlice <<= c->chrSrcVSubSample;

        if (c->vLumFilterPos[i] + c->vLumBufSize < nextSlice)

            c->vLumBufSize = nextSlice - c->vLumFilterPos[i];

        if (c->vChrFilterPos[chrI] + c->vChrBufSize <

            (nextSlice >> c->chrSrcVSubSample))

            c->vChrBufSize = (nextSlice >> c->chrSrcVSubSample) -

                             c->vChrFilterPos[chrI];

    }



    /* Allocate pixbufs (we use dynamic allocation because otherwise we would

     * need to allocate several megabytes to handle all possible cases) */

    FF_ALLOC_OR_GOTO(c, c->lumPixBuf,  c->vLumBufSize * 3 * sizeof(int16_t *), fail);

    FF_ALLOC_OR_GOTO(c, c->chrUPixBuf, c->vChrBufSize * 3 * sizeof(int16_t *), fail);

    FF_ALLOC_OR_GOTO(c, c->chrVPixBuf, c->vChrBufSize * 3 * sizeof(int16_t *), fail);

    if (CONFIG_SWSCALE_ALPHA && isALPHA(c->srcFormat) && isALPHA(c->dstFormat))

        FF_ALLOCZ_OR_GOTO(c, c->alpPixBuf, c->vLumBufSize * 3 * sizeof(int16_t *), fail);

    /* Note we need at least one pixel more at the end because of the MMX code

     * (just in case someone wants to replace the 4000/8000). */

    /* align at 16 bytes for AltiVec */

    for (i = 0; i < c->vLumBufSize; i++) {

        FF_ALLOCZ_OR_GOTO(c, c->lumPixBuf[i + c->vLumBufSize],

                          dst_stride + 16, fail);

        c->lumPixBuf[i] = c->lumPixBuf[i + c->vLumBufSize];

    }

    // 64 / (c->dstBpc & ~7) is the same as 16 / sizeof(scaling_intermediate)

    c->uv_off_px   = dst_stride_px + 64 / (c->dstBpc & ~7);

    c->uv_off_byte = dst_stride + 16;

    for (i = 0; i < c->vChrBufSize; i++) {

        FF_ALLOC_OR_GOTO(c, c->chrUPixBuf[i + c->vChrBufSize],

                         dst_stride * 2 + 32, fail);

        c->chrUPixBuf[i] = c->chrUPixBuf[i + c->vChrBufSize];

        c->chrVPixBuf[i] = c->chrVPixBuf[i + c->vChrBufSize]

                         = c->chrUPixBuf[i] + (dst_stride >> 1) + 8;

    }

    if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf)

        for (i = 0; i < c->vLumBufSize; i++) {

            FF_ALLOCZ_OR_GOTO(c, c->alpPixBuf[i + c->vLumBufSize],

                              dst_stride + 16, fail);

            c->alpPixBuf[i] = c->alpPixBuf[i + c->vLumBufSize];

        }



    // try to avoid drawing green stuff between the right end and the stride end

    for (i = 0; i < c->vChrBufSize; i++)

        memset(c->chrUPixBuf[i], 64, dst_stride * 2 + 1);



    assert(c->chrDstH <= dstH);



    if (flags & SWS_PRINT_INFO) {

        if (flags & SWS_FAST_BILINEAR)

            av_log(c, AV_LOG_INFO, "FAST_BILINEAR scaler, ");

        else if (flags & SWS_BILINEAR)

            av_log(c, AV_LOG_INFO, "BILINEAR scaler, ");

        else if (flags & SWS_BICUBIC)

            av_log(c, AV_LOG_INFO, "BICUBIC scaler, ");

        else if (flags & SWS_X)

            av_log(c, AV_LOG_INFO, "Experimental scaler, ");

        else if (flags & SWS_POINT)

            av_log(c, AV_LOG_INFO, "Nearest Neighbor / POINT scaler, ");

        else if (flags & SWS_AREA)

            av_log(c, AV_LOG_INFO, "Area Averaging scaler, ");

        else if (flags & SWS_BICUBLIN)

            av_log(c, AV_LOG_INFO, "luma BICUBIC / chroma BILINEAR scaler, ");

        else if (flags & SWS_GAUSS)

            av_log(c, AV_LOG_INFO, "Gaussian scaler, ");

        else if (flags & SWS_SINC)

            av_log(c, AV_LOG_INFO, "Sinc scaler, ");

        else if (flags & SWS_LANCZOS)

            av_log(c, AV_LOG_INFO, "Lanczos scaler, ");

        else if (flags & SWS_SPLINE)

            av_log(c, AV_LOG_INFO, "Bicubic spline scaler, ");

        else

            av_log(c, AV_LOG_INFO, "ehh flags invalid?! ");



        av_log(c, AV_LOG_INFO, "from %s to %s%s ",

               sws_format_name(srcFormat),

#ifdef DITHER1XBPP

               dstFormat == AV_PIX_FMT_BGR555   || dstFormat == AV_PIX_FMT_BGR565   ||

               dstFormat == AV_PIX_FMT_RGB444BE || dstFormat == AV_PIX_FMT_RGB444LE ||

               dstFormat == AV_PIX_FMT_BGR444BE || dstFormat == AV_PIX_FMT_BGR444LE ?

                                                             "dithered " : "",

#else

               "",

#endif

               sws_format_name(dstFormat));



        if (INLINE_MMXEXT(cpu_flags))

            av_log(c, AV_LOG_INFO, "using MMXEXT\n");

        else if (INLINE_AMD3DNOW(cpu_flags))

            av_log(c, AV_LOG_INFO, "using 3DNOW\n");

        else if (INLINE_MMX(cpu_flags))

            av_log(c, AV_LOG_INFO, "using MMX\n");

        else if (PPC_ALTIVEC(cpu_flags))

            av_log(c, AV_LOG_INFO, "using AltiVec\n");

        else

            av_log(c, AV_LOG_INFO, "using C\n");



        av_log(c, AV_LOG_VERBOSE, "%dx%d -> %dx%d\n", srcW, srcH, dstW, dstH);

        av_log(c, AV_LOG_DEBUG,

               "lum srcW=%d srcH=%d dstW=%d dstH=%d xInc=%d yInc=%d\n",

               c->srcW, c->srcH, c->dstW, c->dstH, c->lumXInc, c->lumYInc);

        av_log(c, AV_LOG_DEBUG,

               "chr srcW=%d srcH=%d dstW=%d dstH=%d xInc=%d yInc=%d\n",

               c->chrSrcW, c->chrSrcH, c->chrDstW, c->chrDstH,

               c->chrXInc, c->chrYInc);

    }



    c->swscale = ff_getSwsFunc(c);

    return 0;

fail: // FIXME replace things by appropriate error codes

    return -1;

}
