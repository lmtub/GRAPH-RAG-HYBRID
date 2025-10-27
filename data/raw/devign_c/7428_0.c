static void ff_h264_idct8_add_sse2(uint8_t *dst, int16_t *block, int stride)

{

    __asm__ volatile(

        "movdqa   0x10(%1), %%xmm1 \n"

        "movdqa   0x20(%1), %%xmm2 \n"

        "movdqa   0x30(%1), %%xmm3 \n"

        "movdqa   0x50(%1), %%xmm5 \n"

        "movdqa   0x60(%1), %%xmm6 \n"

        "movdqa   0x70(%1), %%xmm7 \n"

        H264_IDCT8_1D_SSE2(%%xmm0, %%xmm1, %%xmm2, %%xmm3, %%xmm4, %%xmm5, %%xmm6, %%xmm7)

        TRANSPOSE8(%%xmm4, %%xmm1, %%xmm7, %%xmm3, %%xmm5, %%xmm0, %%xmm2, %%xmm6, (%1))

        "paddw          %4, %%xmm4 \n"

        "movdqa     %%xmm4, 0x00(%1) \n"

        "movdqa     %%xmm2, 0x40(%1) \n"

        H264_IDCT8_1D_SSE2(%%xmm4, %%xmm0, %%xmm6, %%xmm3, %%xmm2, %%xmm5, %%xmm7, %%xmm1)

        "movdqa     %%xmm6, 0x60(%1) \n"

        "movdqa     %%xmm7, 0x70(%1) \n"

        "pxor       %%xmm7, %%xmm7 \n"

        STORE_DIFF_8P(%%xmm2, (%0),      %%xmm6, %%xmm7)

        STORE_DIFF_8P(%%xmm0, (%0,%2),   %%xmm6, %%xmm7)

        STORE_DIFF_8P(%%xmm1, (%0,%2,2), %%xmm6, %%xmm7)

        STORE_DIFF_8P(%%xmm3, (%0,%3),   %%xmm6, %%xmm7)

        "lea     (%0,%2,4), %0 \n"

        STORE_DIFF_8P(%%xmm5, (%0),      %%xmm6, %%xmm7)

        STORE_DIFF_8P(%%xmm4, (%0,%2),   %%xmm6, %%xmm7)

        "movdqa   0x60(%1), %%xmm0 \n"

        "movdqa   0x70(%1), %%xmm1 \n"

        STORE_DIFF_8P(%%xmm0, (%0,%2,2), %%xmm6, %%xmm7)

        STORE_DIFF_8P(%%xmm1, (%0,%3),   %%xmm6, %%xmm7)

        :"+r"(dst)

        :"r"(block), "r"((x86_reg)stride), "r"((x86_reg)3L*stride), "m"(ff_pw_32)

    );

}
