static av_always_inline void put_h264_qpel8or16_hv1_lowpass_sse2(int16_t *tmp, uint8_t *src, int tmpStride, int srcStride, int size){

    int w = (size+8)>>3;

    src -= 2*srcStride+2;

    while(w--){

        __asm__ volatile(

            "pxor %%xmm7, %%xmm7        \n\t"

            "movq (%0), %%xmm0          \n\t"

            "add %2, %0                 \n\t"

            "movq (%0), %%xmm1          \n\t"

            "add %2, %0                 \n\t"

            "movq (%0), %%xmm2          \n\t"

            "add %2, %0                 \n\t"

            "movq (%0), %%xmm3          \n\t"

            "add %2, %0                 \n\t"

            "movq (%0), %%xmm4          \n\t"

            "add %2, %0                 \n\t"

            "punpcklbw %%xmm7, %%xmm0   \n\t"

            "punpcklbw %%xmm7, %%xmm1   \n\t"

            "punpcklbw %%xmm7, %%xmm2   \n\t"

            "punpcklbw %%xmm7, %%xmm3   \n\t"

            "punpcklbw %%xmm7, %%xmm4   \n\t"

            QPEL_H264HV_XMM(%%xmm0, %%xmm1, %%xmm2, %%xmm3, %%xmm4, %%xmm5, 0*48)

            QPEL_H264HV_XMM(%%xmm1, %%xmm2, %%xmm3, %%xmm4, %%xmm5, %%xmm0, 1*48)

            QPEL_H264HV_XMM(%%xmm2, %%xmm3, %%xmm4, %%xmm5, %%xmm0, %%xmm1, 2*48)

            QPEL_H264HV_XMM(%%xmm3, %%xmm4, %%xmm5, %%xmm0, %%xmm1, %%xmm2, 3*48)

            QPEL_H264HV_XMM(%%xmm4, %%xmm5, %%xmm0, %%xmm1, %%xmm2, %%xmm3, 4*48)

            QPEL_H264HV_XMM(%%xmm5, %%xmm0, %%xmm1, %%xmm2, %%xmm3, %%xmm4, 5*48)

            QPEL_H264HV_XMM(%%xmm0, %%xmm1, %%xmm2, %%xmm3, %%xmm4, %%xmm5, 6*48)

            QPEL_H264HV_XMM(%%xmm1, %%xmm2, %%xmm3, %%xmm4, %%xmm5, %%xmm0, 7*48)

            "cmpl $16, %3               \n\t"

            "jne 2f                     \n\t"

            QPEL_H264HV_XMM(%%xmm2, %%xmm3, %%xmm4, %%xmm5, %%xmm0, %%xmm1,  8*48)

            QPEL_H264HV_XMM(%%xmm3, %%xmm4, %%xmm5, %%xmm0, %%xmm1, %%xmm2,  9*48)

            QPEL_H264HV_XMM(%%xmm4, %%xmm5, %%xmm0, %%xmm1, %%xmm2, %%xmm3, 10*48)

            QPEL_H264HV_XMM(%%xmm5, %%xmm0, %%xmm1, %%xmm2, %%xmm3, %%xmm4, 11*48)

            QPEL_H264HV_XMM(%%xmm0, %%xmm1, %%xmm2, %%xmm3, %%xmm4, %%xmm5, 12*48)

            QPEL_H264HV_XMM(%%xmm1, %%xmm2, %%xmm3, %%xmm4, %%xmm5, %%xmm0, 13*48)

            QPEL_H264HV_XMM(%%xmm2, %%xmm3, %%xmm4, %%xmm5, %%xmm0, %%xmm1, 14*48)

            QPEL_H264HV_XMM(%%xmm3, %%xmm4, %%xmm5, %%xmm0, %%xmm1, %%xmm2, 15*48)

            "2:                         \n\t"

            : "+a"(src)

            : "c"(tmp), "S"((x86_reg)srcStride), "g"(size)

            : XMM_CLOBBERS("%xmm0", "%xmm1", "%xmm2", "%xmm3",

                           "%xmm4", "%xmm5", "%xmm6", "%xmm7",)

              "memory"

        );

        tmp += 8;

        src += 8 - (size+5)*srcStride;

    }

}
