static inline void RENAME(dering)(uint8_t src[], int stride, PPContext *c)

{

#if HAVE_7REGS && (TEMPLATE_PP_MMXEXT || TEMPLATE_PP_3DNOW)

    DECLARE_ALIGNED(8, uint64_t, tmp)[3];

    __asm__ volatile(

        "pxor %%mm6, %%mm6                      \n\t"

        "pcmpeqb %%mm7, %%mm7                   \n\t"

        "movq %2, %%mm0                         \n\t"

        "punpcklbw %%mm6, %%mm0                 \n\t"

        "psrlw $1, %%mm0                        \n\t"

        "psubw %%mm7, %%mm0                     \n\t"

        "packuswb %%mm0, %%mm0                  \n\t"

        "movq %%mm0, %3                         \n\t"



        "lea (%0, %1), %%"REG_a"                \n\t"

        "lea (%%"REG_a", %1, 4), %%"REG_d"      \n\t"



//        0        1        2        3        4        5        6        7        8        9

//        %0        eax        eax+%1        eax+2%1        %0+4%1        edx        edx+%1        edx+2%1        %0+8%1        edx+4%1



#undef REAL_FIND_MIN_MAX

#undef FIND_MIN_MAX

#if TEMPLATE_PP_MMXEXT

#define REAL_FIND_MIN_MAX(addr)\

        "movq " #addr ", %%mm0                  \n\t"\

        "pminub %%mm0, %%mm7                    \n\t"\

        "pmaxub %%mm0, %%mm6                    \n\t"

#else

#define REAL_FIND_MIN_MAX(addr)\

        "movq " #addr ", %%mm0                  \n\t"\

        "movq %%mm7, %%mm1                      \n\t"\

        "psubusb %%mm0, %%mm6                   \n\t"\

        "paddb %%mm0, %%mm6                     \n\t"\

        "psubusb %%mm0, %%mm1                   \n\t"\

        "psubb %%mm1, %%mm7                     \n\t"

#endif

#define FIND_MIN_MAX(addr)  REAL_FIND_MIN_MAX(addr)



FIND_MIN_MAX((%%REGa))

FIND_MIN_MAX((%%REGa, %1))

FIND_MIN_MAX((%%REGa, %1, 2))

FIND_MIN_MAX((%0, %1, 4))

FIND_MIN_MAX((%%REGd))

FIND_MIN_MAX((%%REGd, %1))

FIND_MIN_MAX((%%REGd, %1, 2))

FIND_MIN_MAX((%0, %1, 8))



        "movq %%mm7, %%mm4                      \n\t"

        "psrlq $8, %%mm7                        \n\t"

#if TEMPLATE_PP_MMXEXT

        "pminub %%mm4, %%mm7                    \n\t" // min of pixels

        "pshufw $0xF9, %%mm7, %%mm4             \n\t"

        "pminub %%mm4, %%mm7                    \n\t" // min of pixels

        "pshufw $0xFE, %%mm7, %%mm4             \n\t"

        "pminub %%mm4, %%mm7                    \n\t"

#else

        "movq %%mm7, %%mm1                      \n\t"

        "psubusb %%mm4, %%mm1                   \n\t"

        "psubb %%mm1, %%mm7                     \n\t"

        "movq %%mm7, %%mm4                      \n\t"

        "psrlq $16, %%mm7                       \n\t"

        "movq %%mm7, %%mm1                      \n\t"

        "psubusb %%mm4, %%mm1                   \n\t"

        "psubb %%mm1, %%mm7                     \n\t"

        "movq %%mm7, %%mm4                      \n\t"

        "psrlq $32, %%mm7                       \n\t"

        "movq %%mm7, %%mm1                      \n\t"

        "psubusb %%mm4, %%mm1                   \n\t"

        "psubb %%mm1, %%mm7                     \n\t"

#endif





        "movq %%mm6, %%mm4                      \n\t"

        "psrlq $8, %%mm6                        \n\t"

#if TEMPLATE_PP_MMXEXT

        "pmaxub %%mm4, %%mm6                    \n\t" // max of pixels

        "pshufw $0xF9, %%mm6, %%mm4             \n\t"

        "pmaxub %%mm4, %%mm6                    \n\t"

        "pshufw $0xFE, %%mm6, %%mm4             \n\t"

        "pmaxub %%mm4, %%mm6                    \n\t"

#else

        "psubusb %%mm4, %%mm6                   \n\t"

        "paddb %%mm4, %%mm6                     \n\t"

        "movq %%mm6, %%mm4                      \n\t"

        "psrlq $16, %%mm6                       \n\t"

        "psubusb %%mm4, %%mm6                   \n\t"

        "paddb %%mm4, %%mm6                     \n\t"

        "movq %%mm6, %%mm4                      \n\t"

        "psrlq $32, %%mm6                       \n\t"

        "psubusb %%mm4, %%mm6                   \n\t"

        "paddb %%mm4, %%mm6                     \n\t"

#endif

        "movq %%mm6, %%mm0                      \n\t" // max

        "psubb %%mm7, %%mm6                     \n\t" // max - min

        "push %4                              \n\t"

        "movd %%mm6, %k4                        \n\t"

        "cmpb "MANGLE(deringThreshold)", %b4    \n\t"

        "pop %4                               \n\t"

        " jb 1f                                 \n\t"

        PAVGB(%%mm0, %%mm7)                           // a=(max + min)/2

        "punpcklbw %%mm7, %%mm7                 \n\t"

        "punpcklbw %%mm7, %%mm7                 \n\t"

        "punpcklbw %%mm7, %%mm7                 \n\t"

        "movq %%mm7, (%4)                       \n\t"



        "movq (%0), %%mm0                       \n\t" // L10

        "movq %%mm0, %%mm1                      \n\t" // L10

        "movq %%mm0, %%mm2                      \n\t" // L10

        "psllq $8, %%mm1                        \n\t"

        "psrlq $8, %%mm2                        \n\t"

        "movd -4(%0), %%mm3                     \n\t"

        "movd 8(%0), %%mm4                      \n\t"

        "psrlq $24, %%mm3                       \n\t"

        "psllq $56, %%mm4                       \n\t"

        "por %%mm3, %%mm1                       \n\t" // L00

        "por %%mm4, %%mm2                       \n\t" // L20

        "movq %%mm1, %%mm3                      \n\t" // L00

        PAVGB(%%mm2, %%mm1)                           // (L20 + L00)/2

        PAVGB(%%mm0, %%mm1)                           // (L20 + L00 + 2L10)/4

        "psubusb %%mm7, %%mm0                   \n\t"

        "psubusb %%mm7, %%mm2                   \n\t"

        "psubusb %%mm7, %%mm3                   \n\t"

        "pcmpeqb "MANGLE(b00)", %%mm0           \n\t" // L10 > a ? 0 : -1

        "pcmpeqb "MANGLE(b00)", %%mm2           \n\t" // L20 > a ? 0 : -1

        "pcmpeqb "MANGLE(b00)", %%mm3           \n\t" // L00 > a ? 0 : -1

        "paddb %%mm2, %%mm0                     \n\t"

        "paddb %%mm3, %%mm0                     \n\t"



        "movq (%%"REG_a"), %%mm2                \n\t" // L11

        "movq %%mm2, %%mm3                      \n\t" // L11

        "movq %%mm2, %%mm4                      \n\t" // L11

        "psllq $8, %%mm3                        \n\t"

        "psrlq $8, %%mm4                        \n\t"

        "movd -4(%%"REG_a"), %%mm5              \n\t"

        "movd 8(%%"REG_a"), %%mm6               \n\t"

        "psrlq $24, %%mm5                       \n\t"

        "psllq $56, %%mm6                       \n\t"

        "por %%mm5, %%mm3                       \n\t" // L01

        "por %%mm6, %%mm4                       \n\t" // L21

        "movq %%mm3, %%mm5                      \n\t" // L01

        PAVGB(%%mm4, %%mm3)                           // (L21 + L01)/2

        PAVGB(%%mm2, %%mm3)                           // (L21 + L01 + 2L11)/4

        "psubusb %%mm7, %%mm2                   \n\t"

        "psubusb %%mm7, %%mm4                   \n\t"

        "psubusb %%mm7, %%mm5                   \n\t"

        "pcmpeqb "MANGLE(b00)", %%mm2           \n\t" // L11 > a ? 0 : -1

        "pcmpeqb "MANGLE(b00)", %%mm4           \n\t" // L21 > a ? 0 : -1

        "pcmpeqb "MANGLE(b00)", %%mm5           \n\t" // L01 > a ? 0 : -1

        "paddb %%mm4, %%mm2                     \n\t"

        "paddb %%mm5, %%mm2                     \n\t"

// 0, 2, 3, 1

#define REAL_DERING_CORE(dst,src,ppsx,psx,sx,pplx,plx,lx,t0,t1) \

        "movq " #src ", " #sx "                 \n\t" /* src[0] */\

        "movq " #sx ", " #lx "                  \n\t" /* src[0] */\

        "movq " #sx ", " #t0 "                  \n\t" /* src[0] */\

        "psllq $8, " #lx "                      \n\t"\

        "psrlq $8, " #t0 "                      \n\t"\

        "movd -4" #src ", " #t1 "               \n\t"\

        "psrlq $24, " #t1 "                     \n\t"\

        "por " #t1 ", " #lx "                   \n\t" /* src[-1] */\

        "movd 8" #src ", " #t1 "                \n\t"\

        "psllq $56, " #t1 "                     \n\t"\

        "por " #t1 ", " #t0 "                   \n\t" /* src[+1] */\

        "movq " #lx ", " #t1 "                  \n\t" /* src[-1] */\

        PAVGB(t0, lx)                                 /* (src[-1] + src[+1])/2 */\

        PAVGB(sx, lx)                                 /* (src[-1] + 2src[0] + src[+1])/4 */\

        PAVGB(lx, pplx)                                     \

        "movq " #lx ", 8(%4)                    \n\t"\

        "movq (%4), " #lx "                     \n\t"\

        "psubusb " #lx ", " #t1 "               \n\t"\

        "psubusb " #lx ", " #t0 "               \n\t"\

        "psubusb " #lx ", " #sx "               \n\t"\

        "movq "MANGLE(b00)", " #lx "            \n\t"\

        "pcmpeqb " #lx ", " #t1 "               \n\t" /* src[-1] > a ? 0 : -1*/\

        "pcmpeqb " #lx ", " #t0 "               \n\t" /* src[+1] > a ? 0 : -1*/\

        "pcmpeqb " #lx ", " #sx "               \n\t" /* src[0]  > a ? 0 : -1*/\

        "paddb " #t1 ", " #t0 "                 \n\t"\

        "paddb " #t0 ", " #sx "                 \n\t"\

\

        PAVGB(plx, pplx)                              /* filtered */\

        "movq " #dst ", " #t0 "                 \n\t" /* dst */\

        "movq " #t0 ", " #t1 "                  \n\t" /* dst */\

        "psubusb %3, " #t0 "                    \n\t"\

        "paddusb %3, " #t1 "                    \n\t"\

        PMAXUB(t0, pplx)\

        PMINUB(t1, pplx, t0)\

        "paddb " #sx ", " #ppsx "               \n\t"\

        "paddb " #psx ", " #ppsx "              \n\t"\

        "#paddb "MANGLE(b02)", " #ppsx "        \n\t"\

        "pand "MANGLE(b08)", " #ppsx "          \n\t"\

        "pcmpeqb " #lx ", " #ppsx "             \n\t"\

        "pand " #ppsx ", " #pplx "              \n\t"\

        "pandn " #dst ", " #ppsx "              \n\t"\

        "por " #pplx ", " #ppsx "               \n\t"\

        "movq " #ppsx ", " #dst "               \n\t"\

        "movq 8(%4), " #lx "                    \n\t"



#define DERING_CORE(dst,src,ppsx,psx,sx,pplx,plx,lx,t0,t1) \

   REAL_DERING_CORE(dst,src,ppsx,psx,sx,pplx,plx,lx,t0,t1)

/*

0000000

1111111



1111110

1111101

1111100

1111011

1111010

1111001



1111000

1110111



*/

//DERING_CORE(dst          ,src            ,ppsx ,psx  ,sx   ,pplx ,plx  ,lx   ,t0   ,t1)

DERING_CORE((%%REGa)       ,(%%REGa, %1)   ,%%mm0,%%mm2,%%mm4,%%mm1,%%mm3,%%mm5,%%mm6,%%mm7)

DERING_CORE((%%REGa, %1)   ,(%%REGa, %1, 2),%%mm2,%%mm4,%%mm0,%%mm3,%%mm5,%%mm1,%%mm6,%%mm7)

DERING_CORE((%%REGa, %1, 2),(%0, %1, 4)    ,%%mm4,%%mm0,%%mm2,%%mm5,%%mm1,%%mm3,%%mm6,%%mm7)

DERING_CORE((%0, %1, 4)    ,(%%REGd)       ,%%mm0,%%mm2,%%mm4,%%mm1,%%mm3,%%mm5,%%mm6,%%mm7)

DERING_CORE((%%REGd)       ,(%%REGd, %1)   ,%%mm2,%%mm4,%%mm0,%%mm3,%%mm5,%%mm1,%%mm6,%%mm7)

DERING_CORE((%%REGd, %1)   ,(%%REGd, %1, 2),%%mm4,%%mm0,%%mm2,%%mm5,%%mm1,%%mm3,%%mm6,%%mm7)

DERING_CORE((%%REGd, %1, 2),(%0, %1, 8)    ,%%mm0,%%mm2,%%mm4,%%mm1,%%mm3,%%mm5,%%mm6,%%mm7)

DERING_CORE((%0, %1, 8)    ,(%%REGd, %1, 4),%%mm2,%%mm4,%%mm0,%%mm3,%%mm5,%%mm1,%%mm6,%%mm7)



        "1:                        \n\t"

        : : "r" (src), "r" ((x86_reg)stride), "m" (c->pQPb), "m"(c->pQPb2), "q"(tmp)

        : "%"REG_a, "%"REG_d

    );

#else // HAVE_7REGS && (TEMPLATE_PP_MMXEXT || TEMPLATE_PP_3DNOW)

    int y;

    int min=255;

    int max=0;

    int avg;

    uint8_t *p;

    int s[10];

    const int QP2= c->QP/2 + 1;



    src --;

    for(y=1; y<9; y++){

        int x;

        p= src + stride*y;

        for(x=1; x<9; x++){

            p++;

            if(*p > max) max= *p;

            if(*p < min) min= *p;

        }

    }

    avg= (min + max + 1)>>1;



    if(max - min <deringThreshold) return;



    for(y=0; y<10; y++){

        int t = 0;



        if(src[stride*y + 0] > avg) t+= 1;

        if(src[stride*y + 1] > avg) t+= 2;

        if(src[stride*y + 2] > avg) t+= 4;

        if(src[stride*y + 3] > avg) t+= 8;

        if(src[stride*y + 4] > avg) t+= 16;

        if(src[stride*y + 5] > avg) t+= 32;

        if(src[stride*y + 6] > avg) t+= 64;

        if(src[stride*y + 7] > avg) t+= 128;

        if(src[stride*y + 8] > avg) t+= 256;

        if(src[stride*y + 9] > avg) t+= 512;



        t |= (~t)<<16;

        t &= (t<<1) & (t>>1);

        s[y] = t;

    }



    for(y=1; y<9; y++){

        int t = s[y-1] & s[y] & s[y+1];

        t|= t>>16;

        s[y-1]= t;

    }



    for(y=1; y<9; y++){

        int x;

        int t = s[y-1];



        p= src + stride*y;

        for(x=1; x<9; x++){

            p++;

            if(t & (1<<x)){

                int f= (*(p-stride-1)) + 2*(*(p-stride)) + (*(p-stride+1))

                      +2*(*(p     -1)) + 4*(*p         ) + 2*(*(p     +1))

                      +(*(p+stride-1)) + 2*(*(p+stride)) + (*(p+stride+1));

                f= (f + 8)>>4;



#ifdef DEBUG_DERING_THRESHOLD

                    __asm__ volatile("emms\n\t":);

                    {

                    static long long numPixels=0;

                    if(x!=1 && x!=8 && y!=1 && y!=8) numPixels++;

//                    if((max-min)<20 || (max-min)*QP<200)

//                    if((max-min)*QP < 500)

//                    if(max-min<QP/2)

                    if(max-min < 20){

                        static int numSkipped=0;

                        static int errorSum=0;

                        static int worstQP=0;

                        static int worstRange=0;

                        static int worstDiff=0;

                        int diff= (f - *p);

                        int absDiff= FFABS(diff);

                        int error= diff*diff;



                        if(x==1 || x==8 || y==1 || y==8) continue;



                        numSkipped++;

                        if(absDiff > worstDiff){

                            worstDiff= absDiff;

                            worstQP= QP;

                            worstRange= max-min;

                        }

                        errorSum+= error;



                        if(1024LL*1024LL*1024LL % numSkipped == 0){

                            av_log(c, AV_LOG_INFO, "sum:%1.3f, skip:%d, wQP:%d, "

                                   "wRange:%d, wDiff:%d, relSkip:%1.3f\n",

                                   (float)errorSum/numSkipped, numSkipped, worstQP, worstRange,

                                   worstDiff, (float)numSkipped/numPixels);

                        }

                    }

                    }

#endif

                    if     (*p + QP2 < f) *p= *p + QP2;

                    else if(*p - QP2 > f) *p= *p - QP2;

                    else *p=f;

            }

        }

    }

#ifdef DEBUG_DERING_THRESHOLD

    if(max-min < 20){

        for(y=1; y<9; y++){

            int x;

            int t = 0;

            p= src + stride*y;

            for(x=1; x<9; x++){

                p++;

                *p = FFMIN(*p + 20, 255);

            }

        }

//        src[0] = src[7]=src[stride*7]=src[stride*7 + 7]=255;

    }

#endif

#endif //TEMPLATE_PP_MMXEXT || TEMPLATE_PP_3DNOW

}
