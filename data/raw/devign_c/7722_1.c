static void H264_CHROMA_MC8_TMPL(uint8_t *dst/*align 8*/, uint8_t *src/*align 1*/, int stride, int h, int x, int y, int rnd)

{

    if(y==0 && x==0) {

        /* no filter needed */

        H264_CHROMA_MC8_MV0(dst, src, stride, h);

        return;

    }



    assert(x<8 && y<8 && x>=0 && y>=0);



    if(y==0 || x==0)

    {

        /* 1 dimensional filter only */

        asm volatile(

            "movd %0, %%xmm7 \n\t"

            "movq %1, %%xmm6 \n\t"

            "pshuflw $0, %%xmm7, %%xmm7 \n\t"

            "movlhps %%xmm6, %%xmm6 \n\t"

            "movlhps %%xmm7, %%xmm7 \n\t"

            :: "r"(255*(x+y)+8), "m"(rnd?ff_pw_4:ff_pw_3));



        if(x) {

            asm volatile(

                "1: \n\t"

                "movq (%1), %%xmm0 \n\t"

                "movq 1(%1), %%xmm1 \n\t"

                "movq (%1,%3), %%xmm2 \n\t"

                "movq 1(%1,%3), %%xmm3 \n\t"

                "punpcklbw %%xmm1, %%xmm0 \n\t"

                "punpcklbw %%xmm3, %%xmm2 \n\t"

                "pmaddubsw %%xmm7, %%xmm0 \n\t"

                "pmaddubsw %%xmm7, %%xmm2 \n\t"

         AVG_OP("movq (%0), %%xmm4 \n\t")

         AVG_OP("movhps (%0,%3), %%xmm4 \n\t")

                "paddw %%xmm6, %%xmm0 \n\t"

                "paddw %%xmm6, %%xmm2 \n\t"

                "psrlw $3, %%xmm0 \n\t"

                "psrlw $3, %%xmm2 \n\t"

                "packuswb %%xmm2, %%xmm0 \n\t"

         AVG_OP("pavgb %%xmm4, %%xmm0 \n\t")

                "movq %%xmm0, (%0) \n\t"

                "movhps %%xmm0, (%0,%3) \n\t"

                "sub $2, %2 \n\t"

                "lea (%1,%3,2), %1 \n\t"

                "lea (%0,%3,2), %0 \n\t"

                "jg 1b \n\t"

                :"+r"(dst), "+r"(src), "+r"(h)

                :"r"((long)stride)

            );

        } else {

            asm volatile(

                "1: \n\t"

                "movq (%1), %%xmm0 \n\t"

                "movq (%1,%3), %%xmm1 \n\t"

                "movdqa %%xmm1, %%xmm2 \n\t"

                "movq (%1,%3,2), %%xmm3 \n\t"

                "punpcklbw %%xmm1, %%xmm0 \n\t"

                "punpcklbw %%xmm3, %%xmm2 \n\t"

                "pmaddubsw %%xmm7, %%xmm0 \n\t"

                "pmaddubsw %%xmm7, %%xmm2 \n\t"

         AVG_OP("movq (%0), %%xmm4 \n\t")

         AVG_OP("movhps (%0,%3), %%xmm4 \n\t")

                "paddw %%xmm6, %%xmm0 \n\t"

                "paddw %%xmm6, %%xmm2 \n\t"

                "psrlw $3, %%xmm0 \n\t"

                "psrlw $3, %%xmm2 \n\t"

                "packuswb %%xmm2, %%xmm0 \n\t"

         AVG_OP("pavgb %%xmm4, %%xmm0 \n\t")

                "movq %%xmm0, (%0) \n\t"

                "movhps %%xmm0, (%0,%3) \n\t"

                "sub $2, %2 \n\t"

                "lea (%1,%3,2), %1 \n\t"

                "lea (%0,%3,2), %0 \n\t"

                "jg 1b \n\t"

                :"+r"(dst), "+r"(src), "+r"(h)

                :"r"((long)stride)

            );

        }

        return;

    }



    /* general case, bilinear */

    asm volatile(

        "movd %0, %%xmm7 \n\t"

        "movd %1, %%xmm6 \n\t"

        "movdqa %2, %%xmm5 \n\t"

        "pshuflw $0, %%xmm7, %%xmm7 \n\t"

        "pshuflw $0, %%xmm6, %%xmm6 \n\t"

        "movlhps %%xmm7, %%xmm7 \n\t"

        "movlhps %%xmm6, %%xmm6 \n\t"

        :: "r"((x*255+8)*(8-y)), "r"((x*255+8)*y), "m"(rnd?ff_pw_32:ff_pw_28)

    );



    asm volatile(

        "movq (%1), %%xmm0 \n\t"

        "movq 1(%1), %%xmm1 \n\t"

        "punpcklbw %%xmm1, %%xmm0 \n\t"

        "add %3, %1 \n\t"

        "1: \n\t"

        "movq (%1), %%xmm1 \n\t"

        "movq 1(%1), %%xmm2 \n\t"

        "movq (%1,%3), %%xmm3 \n\t"

        "movq 1(%1,%3), %%xmm4 \n\t"

        "lea (%1,%3,2), %1 \n\t"

        "punpcklbw %%xmm2, %%xmm1 \n\t"

        "punpcklbw %%xmm4, %%xmm3 \n\t"

        "movdqa %%xmm1, %%xmm2 \n\t"

        "movdqa %%xmm3, %%xmm4 \n\t"

        "pmaddubsw %%xmm7, %%xmm0 \n\t"

        "pmaddubsw %%xmm6, %%xmm1 \n\t"

        "pmaddubsw %%xmm7, %%xmm2 \n\t"

        "pmaddubsw %%xmm6, %%xmm3 \n\t"

        "paddw %%xmm5, %%xmm0 \n\t"

        "paddw %%xmm5, %%xmm2 \n\t"

        "paddw %%xmm0, %%xmm1 \n\t"

        "paddw %%xmm2, %%xmm3 \n\t"

        "movdqa %%xmm4, %%xmm0 \n\t"

        "psrlw $6, %%xmm1 \n\t"

        "psrlw $6, %%xmm3 \n\t"

 AVG_OP("movq (%0), %%xmm2 \n\t")

 AVG_OP("movhps (%0,%3), %%xmm2 \n\t")

        "packuswb %%xmm3, %%xmm1 \n\t"

 AVG_OP("pavgb %%xmm2, %%xmm1 \n\t")

        "movq %%xmm1, (%0)\n\t"

        "movhps %%xmm1, (%0,%3)\n\t"

        "sub $2, %2 \n\t"

        "lea (%0,%3,2), %0 \n\t"

        "jg 1b \n\t"

        :"+r"(dst), "+r"(src), "+r"(h)

        :"r"((long)stride)

    );

}
