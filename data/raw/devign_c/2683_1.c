static inline void RENAME(yuv2packed2)(SwsContext *c, uint16_t *buf0, uint16_t *buf1, uint16_t *uvbuf0, uint16_t *uvbuf1,

                          uint8_t *dest, int dstW, int yalpha, int uvalpha, int y)

{

    int yalpha1=yalpha^4095;

    int uvalpha1=uvalpha^4095;

    int i;



#if 0 //isn't used

    if (flags&SWS_FULL_CHR_H_INT)

    {

        switch(dstFormat)

        {

#ifdef HAVE_MMX

        case PIX_FMT_RGB32:

            asm volatile(





FULL_YSCALEYUV2RGB

            "punpcklbw %%mm1, %%mm3    \n\t" // BGBGBGBG

            "punpcklbw %%mm7, %%mm0    \n\t" // R0R0R0R0



            "movq      %%mm3, %%mm1    \n\t"

            "punpcklwd %%mm0, %%mm3    \n\t" // BGR0BGR0

            "punpckhwd %%mm0, %%mm1    \n\t" // BGR0BGR0



            MOVNTQ(%%mm3,  (%4, %%REGa, 4))

            MOVNTQ(%%mm1, 8(%4, %%REGa, 4))



            "add $4, %%"REG_a"  \n\t"

            "cmp %5, %%"REG_a"  \n\t"

            " jb 1b             \n\t"



            :: "r" (buf0), "r" (buf1), "r" (uvbuf0), "r" (uvbuf1), "r" (dest), "m" ((long)dstW),

            "m" (yalpha1), "m" (uvalpha1)

            : "%"REG_a

            );

            break;

        case PIX_FMT_BGR24:

            asm volatile(



FULL_YSCALEYUV2RGB



                                              // lsb ... msb

            "punpcklbw %%mm1, %%mm3     \n\t" // BGBGBGBG

            "punpcklbw %%mm7, %%mm0     \n\t" // R0R0R0R0



            "movq      %%mm3, %%mm1     \n\t"

            "punpcklwd %%mm0, %%mm3     \n\t" // BGR0BGR0

            "punpckhwd %%mm0, %%mm1     \n\t" // BGR0BGR0



            "movq      %%mm3, %%mm2     \n\t" // BGR0BGR0

            "psrlq        $8, %%mm3     \n\t" // GR0BGR00

            "pand "MANGLE(bm00000111)", %%mm2   \n\t" // BGR00000

            "pand "MANGLE(bm11111000)", %%mm3   \n\t" // 000BGR00

            "por       %%mm2, %%mm3     \n\t" // BGRBGR00

            "movq      %%mm1, %%mm2     \n\t"

            "psllq       $48, %%mm1     \n\t" // 000000BG

            "por       %%mm1, %%mm3     \n\t" // BGRBGRBG



            "movq      %%mm2, %%mm1     \n\t" // BGR0BGR0

            "psrld       $16, %%mm2     \n\t" // R000R000

            "psrlq       $24, %%mm1     \n\t" // 0BGR0000

            "por       %%mm2, %%mm1     \n\t" // RBGRR000



            "mov          %4, %%"REG_b" \n\t"

            "add   %%"REG_a", %%"REG_b" \n\t"



#ifdef HAVE_MMX2

            //FIXME Alignment

            "movntq %%mm3,  (%%"REG_b", %%"REG_a", 2)   \n\t"

            "movntq %%mm1, 8(%%"REG_b", %%"REG_a", 2)   \n\t"

#else

            "movd %%mm3,  (%%"REG_b", %%"REG_a", 2)     \n\t"

            "psrlq  $32, %%mm3                          \n\t"

            "movd %%mm3, 4(%%"REG_b", %%"REG_a", 2)     \n\t"

            "movd %%mm1, 8(%%"REG_b", %%"REG_a", 2)     \n\t"

#endif

            "add     $4, %%"REG_a"                      \n\t"

            "cmp     %5, %%"REG_a"                      \n\t"

            " jb     1b                                 \n\t"



            :: "r" (buf0), "r" (buf1), "r" (uvbuf0), "r" (uvbuf1), "m" (dest), "m" (dstW),

            "m" (yalpha1), "m" (uvalpha1)

            : "%"REG_a, "%"REG_b

            );

            break;

        case PIX_FMT_BGR555:

            asm volatile(



FULL_YSCALEYUV2RGB

#ifdef DITHER1XBPP

            "paddusb "MANGLE(g5Dither)", %%mm1  \n\t"

            "paddusb "MANGLE(r5Dither)", %%mm0  \n\t"

            "paddusb "MANGLE(b5Dither)", %%mm3  \n\t"

#endif

            "punpcklbw            %%mm7, %%mm1  \n\t" // 0G0G0G0G

            "punpcklbw            %%mm7, %%mm3  \n\t" // 0B0B0B0B

            "punpcklbw            %%mm7, %%mm0  \n\t" // 0R0R0R0R



            "psrlw                   $3, %%mm3  \n\t"

            "psllw                   $2, %%mm1  \n\t"

            "psllw                   $7, %%mm0  \n\t"

            "pand     "MANGLE(g15Mask)", %%mm1  \n\t"

            "pand     "MANGLE(r15Mask)", %%mm0  \n\t"



            "por                  %%mm3, %%mm1  \n\t"

            "por                  %%mm1, %%mm0  \n\t"



            MOVNTQ(%%mm0, (%4, %%REGa, 2))



            "add $4, %%"REG_a"  \n\t"

            "cmp %5, %%"REG_a"  \n\t"

            " jb 1b             \n\t"



            :: "r" (buf0), "r" (buf1), "r" (uvbuf0), "r" (uvbuf1), "r" (dest), "m" (dstW),

            "m" (yalpha1), "m" (uvalpha1)

            : "%"REG_a

            );

            break;

        case PIX_FMT_BGR565:

            asm volatile(



FULL_YSCALEYUV2RGB

#ifdef DITHER1XBPP

            "paddusb "MANGLE(g6Dither)", %%mm1  \n\t"

            "paddusb "MANGLE(r5Dither)", %%mm0  \n\t"

            "paddusb "MANGLE(b5Dither)", %%mm3  \n\t"

#endif

            "punpcklbw            %%mm7, %%mm1  \n\t" // 0G0G0G0G

            "punpcklbw            %%mm7, %%mm3  \n\t" // 0B0B0B0B

            "punpcklbw            %%mm7, %%mm0  \n\t" // 0R0R0R0R



            "psrlw                   $3, %%mm3  \n\t"

            "psllw                   $3, %%mm1  \n\t"

            "psllw                   $8, %%mm0  \n\t"

            "pand     "MANGLE(g16Mask)", %%mm1  \n\t"

            "pand     "MANGLE(r16Mask)", %%mm0  \n\t"



            "por                  %%mm3, %%mm1  \n\t"

            "por                  %%mm1, %%mm0  \n\t"



            MOVNTQ(%%mm0, (%4, %%REGa, 2))



            "add $4, %%"REG_a"  \n\t"

            "cmp %5, %%"REG_a"  \n\t"

            " jb 1b             \n\t"



            :: "r" (buf0), "r" (buf1), "r" (uvbuf0), "r" (uvbuf1), "r" (dest), "m" (dstW),

            "m" (yalpha1), "m" (uvalpha1)

            : "%"REG_a

            );

            break;

#endif /* HAVE_MMX */

        case PIX_FMT_BGR32:

#ifndef HAVE_MMX

        case PIX_FMT_RGB32:

#endif

            if (dstFormat==PIX_FMT_RGB32)

            {

                int i;

#ifdef WORDS_BIGENDIAN

                dest++;

#endif

                for (i=0;i<dstW;i++){

                    // vertical linear interpolation && yuv2rgb in a single step:

                    int Y=yuvtab_2568[((buf0[i]*yalpha1+buf1[i]*yalpha)>>19)];

                    int U=((uvbuf0[i]*uvalpha1+uvbuf1[i]*uvalpha)>>19);

                    int V=((uvbuf0[i+2048]*uvalpha1+uvbuf1[i+2048]*uvalpha)>>19);

                    dest[0]=clip_table[((Y + yuvtab_40cf[U]) >>13)];

                    dest[1]=clip_table[((Y + yuvtab_1a1e[V] + yuvtab_0c92[U]) >>13)];

                    dest[2]=clip_table[((Y + yuvtab_3343[V]) >>13)];

                    dest+= 4;

                }

            }

            else if (dstFormat==PIX_FMT_BGR24)

            {

                int i;

                for (i=0;i<dstW;i++){

                    // vertical linear interpolation && yuv2rgb in a single step:

                    int Y=yuvtab_2568[((buf0[i]*yalpha1+buf1[i]*yalpha)>>19)];

                    int U=((uvbuf0[i]*uvalpha1+uvbuf1[i]*uvalpha)>>19);

                    int V=((uvbuf0[i+2048]*uvalpha1+uvbuf1[i+2048]*uvalpha)>>19);

                    dest[0]=clip_table[((Y + yuvtab_40cf[U]) >>13)];

                    dest[1]=clip_table[((Y + yuvtab_1a1e[V] + yuvtab_0c92[U]) >>13)];

                    dest[2]=clip_table[((Y + yuvtab_3343[V]) >>13)];

                    dest+= 3;

                }

            }

            else if (dstFormat==PIX_FMT_BGR565)

            {

                int i;

                for (i=0;i<dstW;i++){

                    // vertical linear interpolation && yuv2rgb in a single step:

                    int Y=yuvtab_2568[((buf0[i]*yalpha1+buf1[i]*yalpha)>>19)];

                    int U=((uvbuf0[i]*uvalpha1+uvbuf1[i]*uvalpha)>>19);

                    int V=((uvbuf0[i+2048]*uvalpha1+uvbuf1[i+2048]*uvalpha)>>19);



                    ((uint16_t*)dest)[i] =

                        clip_table16b[(Y + yuvtab_40cf[U]) >>13] |

                        clip_table16g[(Y + yuvtab_1a1e[V] + yuvtab_0c92[U]) >>13] |

                        clip_table16r[(Y + yuvtab_3343[V]) >>13];

                }

            }

            else if (dstFormat==PIX_FMT_BGR555)

            {

                int i;

                for (i=0;i<dstW;i++){

                    // vertical linear interpolation && yuv2rgb in a single step:

                    int Y=yuvtab_2568[((buf0[i]*yalpha1+buf1[i]*yalpha)>>19)];

                    int U=((uvbuf0[i]*uvalpha1+uvbuf1[i]*uvalpha)>>19);

                    int V=((uvbuf0[i+2048]*uvalpha1+uvbuf1[i+2048]*uvalpha)>>19);



                    ((uint16_t*)dest)[i] =

                        clip_table15b[(Y + yuvtab_40cf[U]) >>13] |

                        clip_table15g[(Y + yuvtab_1a1e[V] + yuvtab_0c92[U]) >>13] |

                        clip_table15r[(Y + yuvtab_3343[V]) >>13];

                }

            }

        }//FULL_UV_IPOL

    else

    {

#endif // if 0

#ifdef HAVE_MMX

        switch(c->dstFormat)

        {

            //Note 8280 == DSTW_OFFSET but the preprocessor can't handle that there :(

            case PIX_FMT_RGB32:

                asm volatile(

                "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                "mov        %4, %%"REG_b"               \n\t"

                "push %%"REG_BP"                        \n\t"

                YSCALEYUV2RGB(%%REGBP, %5)

                WRITEBGR32(%%REGb, 8280(%5), %%REGBP)

                "pop %%"REG_BP"                         \n\t"

                "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                "a" (&c->redDither)

                );

                return;

            case PIX_FMT_BGR24:

                asm volatile(

                "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                "mov        %4, %%"REG_b"               \n\t"

                "push %%"REG_BP"                        \n\t"

                YSCALEYUV2RGB(%%REGBP, %5)

                WRITEBGR24(%%REGb, 8280(%5), %%REGBP)

                "pop %%"REG_BP"                         \n\t"

                "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"

                :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                "a" (&c->redDither)

                );

                return;

            case PIX_FMT_BGR555:

                asm volatile(

                "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                "mov        %4, %%"REG_b"               \n\t"

                "push %%"REG_BP"                        \n\t"

                YSCALEYUV2RGB(%%REGBP, %5)

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

                "paddusb "MANGLE(b5Dither)", %%mm2      \n\t"

                "paddusb "MANGLE(g5Dither)", %%mm4      \n\t"

                "paddusb "MANGLE(r5Dither)", %%mm5      \n\t"

#endif



                WRITEBGR15(%%REGb, 8280(%5), %%REGBP)

                "pop %%"REG_BP"                         \n\t"

                "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                "a" (&c->redDither)

                );

                return;

            case PIX_FMT_BGR565:

                asm volatile(

                "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                "mov        %4, %%"REG_b"               \n\t"

                "push %%"REG_BP"                        \n\t"

                YSCALEYUV2RGB(%%REGBP, %5)

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

                "paddusb "MANGLE(b5Dither)", %%mm2      \n\t"

                "paddusb "MANGLE(g6Dither)", %%mm4      \n\t"

                "paddusb "MANGLE(r5Dither)", %%mm5      \n\t"

#endif



                WRITEBGR16(%%REGb, 8280(%5), %%REGBP)

                "pop %%"REG_BP"                         \n\t"

                "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"

                :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                "a" (&c->redDither)

                );

                return;

            case PIX_FMT_YUYV422:

                asm volatile(

                "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                "mov %4, %%"REG_b"                        \n\t"

                "push %%"REG_BP"                        \n\t"

                YSCALEYUV2PACKED(%%REGBP, %5)

                WRITEYUY2(%%REGb, 8280(%5), %%REGBP)

                "pop %%"REG_BP"                         \n\t"

                "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"

                :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                "a" (&c->redDither)

                );

                return;

            default: break;

        }

#endif //HAVE_MMX

YSCALE_YUV_2_ANYRGB_C(YSCALE_YUV_2_RGB2_C, YSCALE_YUV_2_PACKED2_C)

}
