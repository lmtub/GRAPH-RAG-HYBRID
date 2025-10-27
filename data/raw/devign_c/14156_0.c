PAETH(mmx2, ABS3_MMX2)

#ifdef HAVE_SSSE3

PAETH(ssse3, ABS3_SSSE3)

#endif



#define QPEL_V_LOW(m3,m4,m5,m6, pw_20, pw_3, rnd, in0, in1, in2, in7, out, OP)\

        "paddw " #m4 ", " #m3 "           \n\t" /* x1 */\

        "movq "MANGLE(ff_pw_20)", %%mm4   \n\t" /* 20 */\

        "pmullw " #m3 ", %%mm4            \n\t" /* 20x1 */\

        "movq "#in7", " #m3 "             \n\t" /* d */\

        "movq "#in0", %%mm5               \n\t" /* D */\

        "paddw " #m3 ", %%mm5             \n\t" /* x4 */\

        "psubw %%mm5, %%mm4               \n\t" /* 20x1 - x4 */\

        "movq "#in1", %%mm5               \n\t" /* C */\

        "movq "#in2", %%mm6               \n\t" /* B */\

        "paddw " #m6 ", %%mm5             \n\t" /* x3 */\

        "paddw " #m5 ", %%mm6             \n\t" /* x2 */\

        "paddw %%mm6, %%mm6               \n\t" /* 2x2 */\

        "psubw %%mm6, %%mm5               \n\t" /* -2x2 + x3 */\

        "pmullw "MANGLE(ff_pw_3)", %%mm5  \n\t" /* -6x2 + 3x3 */\

        "paddw " #rnd ", %%mm4            \n\t" /* x2 */\

        "paddw %%mm4, %%mm5               \n\t" /* 20x1 - 6x2 + 3x3 - x4 */\

        "psraw $5, %%mm5                  \n\t"\

        "packuswb %%mm5, %%mm5            \n\t"\

        OP(%%mm5, out, %%mm7, d)



#define QPEL_BASE(OPNAME, ROUNDER, RND, OP_MMX2, OP_3DNOW)\

static void OPNAME ## mpeg4_qpel16_h_lowpass_mmx2(uint8_t *dst, uint8_t *src, int dstStride, int srcStride, int h){\

    uint64_t temp;\

\

    asm volatile(\

        "pxor %%mm7, %%mm7                \n\t"\

        "1:                               \n\t"\

        "movq  (%0), %%mm0                \n\t" /* ABCDEFGH */\

        "movq %%mm0, %%mm1                \n\t" /* ABCDEFGH */\

        "movq %%mm0, %%mm2                \n\t" /* ABCDEFGH */\

        "punpcklbw %%mm7, %%mm0           \n\t" /* 0A0B0C0D */\

        "punpckhbw %%mm7, %%mm1           \n\t" /* 0E0F0G0H */\

        "pshufw $0x90, %%mm0, %%mm5       \n\t" /* 0A0A0B0C */\

        "pshufw $0x41, %%mm0, %%mm6       \n\t" /* 0B0A0A0B */\

        "movq %%mm2, %%mm3                \n\t" /* ABCDEFGH */\

        "movq %%mm2, %%mm4                \n\t" /* ABCDEFGH */\

        "psllq $8, %%mm2                  \n\t" /* 0ABCDEFG */\

        "psllq $16, %%mm3                 \n\t" /* 00ABCDEF */\

        "psllq $24, %%mm4                 \n\t" /* 000ABCDE */\

        "punpckhbw %%mm7, %%mm2           \n\t" /* 0D0E0F0G */\

        "punpckhbw %%mm7, %%mm3           \n\t" /* 0C0D0E0F */\

        "punpckhbw %%mm7, %%mm4           \n\t" /* 0B0C0D0E */\

        "paddw %%mm3, %%mm5               \n\t" /* b */\

        "paddw %%mm2, %%mm6               \n\t" /* c */\

        "paddw %%mm5, %%mm5               \n\t" /* 2b */\

        "psubw %%mm5, %%mm6               \n\t" /* c - 2b */\

        "pshufw $0x06, %%mm0, %%mm5       \n\t" /* 0C0B0A0A */\

        "pmullw "MANGLE(ff_pw_3)", %%mm6  \n\t" /* 3c - 6b */\

        "paddw %%mm4, %%mm0               \n\t" /* a */\

        "paddw %%mm1, %%mm5               \n\t" /* d */\

        "pmullw "MANGLE(ff_pw_20)", %%mm0 \n\t" /* 20a */\

        "psubw %%mm5, %%mm0               \n\t" /* 20a - d */\

        "paddw %6, %%mm6                  \n\t"\

        "paddw %%mm6, %%mm0               \n\t" /* 20a - 6b + 3c - d */\

        "psraw $5, %%mm0                  \n\t"\

        "movq %%mm0, %5                   \n\t"\

        /* mm1=EFGH, mm2=DEFG, mm3=CDEF, mm4=BCDE, mm7=0 */\

        \

        "movq 5(%0), %%mm0                \n\t" /* FGHIJKLM */\

        "movq %%mm0, %%mm5                \n\t" /* FGHIJKLM */\

        "movq %%mm0, %%mm6                \n\t" /* FGHIJKLM */\

        "psrlq $8, %%mm0                  \n\t" /* GHIJKLM0 */\

        "psrlq $16, %%mm5                 \n\t" /* HIJKLM00 */\

        "punpcklbw %%mm7, %%mm0           \n\t" /* 0G0H0I0J */\

        "punpcklbw %%mm7, %%mm5           \n\t" /* 0H0I0J0K */\

        "paddw %%mm0, %%mm2               \n\t" /* b */\

        "paddw %%mm5, %%mm3               \n\t" /* c */\

        "paddw %%mm2, %%mm2               \n\t" /* 2b */\

        "psubw %%mm2, %%mm3               \n\t" /* c - 2b */\

        "movq %%mm6, %%mm2                \n\t" /* FGHIJKLM */\

        "psrlq $24, %%mm6                 \n\t" /* IJKLM000 */\

        "punpcklbw %%mm7, %%mm2           \n\t" /* 0F0G0H0I */\

        "punpcklbw %%mm7, %%mm6           \n\t" /* 0I0J0K0L */\

        "pmullw "MANGLE(ff_pw_3)", %%mm3  \n\t" /* 3c - 6b */\

        "paddw %%mm2, %%mm1               \n\t" /* a */\

        "paddw %%mm6, %%mm4               \n\t" /* d */\

        "pmullw "MANGLE(ff_pw_20)", %%mm1 \n\t" /* 20a */\

        "psubw %%mm4, %%mm3               \n\t" /* - 6b +3c - d */\

        "paddw %6, %%mm1                  \n\t"\

        "paddw %%mm1, %%mm3               \n\t" /* 20a - 6b +3c - d */\

        "psraw $5, %%mm3                  \n\t"\

        "movq %5, %%mm1                   \n\t"\

        "packuswb %%mm3, %%mm1            \n\t"\

        OP_MMX2(%%mm1, (%1),%%mm4, q)\

        /* mm0= GHIJ, mm2=FGHI, mm5=HIJK, mm6=IJKL, mm7=0 */\

        \

        "movq 9(%0), %%mm1                \n\t" /* JKLMNOPQ */\

        "movq %%mm1, %%mm4                \n\t" /* JKLMNOPQ */\

        "movq %%mm1, %%mm3                \n\t" /* JKLMNOPQ */\

        "psrlq $8, %%mm1                  \n\t" /* KLMNOPQ0 */\

        "psrlq $16, %%mm4                 \n\t" /* LMNOPQ00 */\

        "punpcklbw %%mm7, %%mm1           \n\t" /* 0K0L0M0N */\

        "punpcklbw %%mm7, %%mm4           \n\t" /* 0L0M0N0O */\

        "paddw %%mm1, %%mm5               \n\t" /* b */\

        "paddw %%mm4, %%mm0               \n\t" /* c */\

        "paddw %%mm5, %%mm5               \n\t" /* 2b */\

        "psubw %%mm5, %%mm0               \n\t" /* c - 2b */\

        "movq %%mm3, %%mm5                \n\t" /* JKLMNOPQ */\

        "psrlq $24, %%mm3                 \n\t" /* MNOPQ000 */\

        "pmullw "MANGLE(ff_pw_3)", %%mm0  \n\t" /* 3c - 6b */\

        "punpcklbw %%mm7, %%mm3           \n\t" /* 0M0N0O0P */\

        "paddw %%mm3, %%mm2               \n\t" /* d */\

        "psubw %%mm2, %%mm0               \n\t" /* -6b + 3c - d */\

        "movq %%mm5, %%mm2                \n\t" /* JKLMNOPQ */\

        "punpcklbw %%mm7, %%mm2           \n\t" /* 0J0K0L0M */\

        "punpckhbw %%mm7, %%mm5           \n\t" /* 0N0O0P0Q */\

        "paddw %%mm2, %%mm6               \n\t" /* a */\

        "pmullw "MANGLE(ff_pw_20)", %%mm6 \n\t" /* 20a */\

        "paddw %6, %%mm0                  \n\t"\

        "paddw %%mm6, %%mm0               \n\t" /* 20a - 6b + 3c - d */\

        "psraw $5, %%mm0                  \n\t"\

        /* mm1=KLMN, mm2=JKLM, mm3=MNOP, mm4=LMNO, mm5=NOPQ mm7=0 */\

        \

        "paddw %%mm5, %%mm3               \n\t" /* a */\

        "pshufw $0xF9, %%mm5, %%mm6       \n\t" /* 0O0P0Q0Q */\

        "paddw %%mm4, %%mm6               \n\t" /* b */\

        "pshufw $0xBE, %%mm5, %%mm4       \n\t" /* 0P0Q0Q0P */\

        "pshufw $0x6F, %%mm5, %%mm5       \n\t" /* 0Q0Q0P0O */\

        "paddw %%mm1, %%mm4               \n\t" /* c */\

        "paddw %%mm2, %%mm5               \n\t" /* d */\

        "paddw %%mm6, %%mm6               \n\t" /* 2b */\

        "psubw %%mm6, %%mm4               \n\t" /* c - 2b */\

        "pmullw "MANGLE(ff_pw_20)", %%mm3 \n\t" /* 20a */\

        "pmullw "MANGLE(ff_pw_3)", %%mm4  \n\t" /* 3c - 6b */\

        "psubw %%mm5, %%mm3               \n\t" /* -6b + 3c - d */\

        "paddw %6, %%mm4                  \n\t"\

        "paddw %%mm3, %%mm4               \n\t" /* 20a - 6b + 3c - d */\

        "psraw $5, %%mm4                  \n\t"\

        "packuswb %%mm4, %%mm0            \n\t"\

        OP_MMX2(%%mm0, 8(%1), %%mm4, q)\

        \

        "add %3, %0                       \n\t"\

        "add %4, %1                       \n\t"\

        "decl %2                          \n\t"\

        " jnz 1b                          \n\t"\

        : "+a"(src), "+c"(dst), "+g"(h)\

        : "d"((long)srcStride), "S"((long)dstStride), /*"m"(ff_pw_20), "m"(ff_pw_3),*/ "m"(temp), "m"(ROUNDER)\

        : "memory"\

    );\

}\

\

static void OPNAME ## mpeg4_qpel16_h_lowpass_3dnow(uint8_t *dst, uint8_t *src, int dstStride, int srcStride, int h){\

    int i;\

    int16_t temp[16];\

    /* quick HACK, XXX FIXME MUST be optimized */\

    for(i=0; i<h; i++)\

    {\

        temp[ 0]= (src[ 0]+src[ 1])*20 - (src[ 0]+src[ 2])*6 + (src[ 1]+src[ 3])*3 - (src[ 2]+src[ 4]);\

        temp[ 1]= (src[ 1]+src[ 2])*20 - (src[ 0]+src[ 3])*6 + (src[ 0]+src[ 4])*3 - (src[ 1]+src[ 5]);\

        temp[ 2]= (src[ 2]+src[ 3])*20 - (src[ 1]+src[ 4])*6 + (src[ 0]+src[ 5])*3 - (src[ 0]+src[ 6]);\

        temp[ 3]= (src[ 3]+src[ 4])*20 - (src[ 2]+src[ 5])*6 + (src[ 1]+src[ 6])*3 - (src[ 0]+src[ 7]);\

        temp[ 4]= (src[ 4]+src[ 5])*20 - (src[ 3]+src[ 6])*6 + (src[ 2]+src[ 7])*3 - (src[ 1]+src[ 8]);\

        temp[ 5]= (src[ 5]+src[ 6])*20 - (src[ 4]+src[ 7])*6 + (src[ 3]+src[ 8])*3 - (src[ 2]+src[ 9]);\

        temp[ 6]= (src[ 6]+src[ 7])*20 - (src[ 5]+src[ 8])*6 + (src[ 4]+src[ 9])*3 - (src[ 3]+src[10]);\

        temp[ 7]= (src[ 7]+src[ 8])*20 - (src[ 6]+src[ 9])*6 + (src[ 5]+src[10])*3 - (src[ 4]+src[11]);\

        temp[ 8]= (src[ 8]+src[ 9])*20 - (src[ 7]+src[10])*6 + (src[ 6]+src[11])*3 - (src[ 5]+src[12]);\

        temp[ 9]= (src[ 9]+src[10])*20 - (src[ 8]+src[11])*6 + (src[ 7]+src[12])*3 - (src[ 6]+src[13]);\

        temp[10]= (src[10]+src[11])*20 - (src[ 9]+src[12])*6 + (src[ 8]+src[13])*3 - (src[ 7]+src[14]);\

        temp[11]= (src[11]+src[12])*20 - (src[10]+src[13])*6 + (src[ 9]+src[14])*3 - (src[ 8]+src[15]);\

        temp[12]= (src[12]+src[13])*20 - (src[11]+src[14])*6 + (src[10]+src[15])*3 - (src[ 9]+src[16]);\

        temp[13]= (src[13]+src[14])*20 - (src[12]+src[15])*6 + (src[11]+src[16])*3 - (src[10]+src[16]);\

        temp[14]= (src[14]+src[15])*20 - (src[13]+src[16])*6 + (src[12]+src[16])*3 - (src[11]+src[15]);\

        temp[15]= (src[15]+src[16])*20 - (src[14]+src[16])*6 + (src[13]+src[15])*3 - (src[12]+src[14]);\

        asm volatile(\

            "movq (%0), %%mm0               \n\t"\

            "movq 8(%0), %%mm1              \n\t"\

            "paddw %2, %%mm0                \n\t"\

            "paddw %2, %%mm1                \n\t"\

            "psraw $5, %%mm0                \n\t"\

            "psraw $5, %%mm1                \n\t"\

            "packuswb %%mm1, %%mm0          \n\t"\

            OP_3DNOW(%%mm0, (%1), %%mm1, q)\

            "movq 16(%0), %%mm0             \n\t"\

            "movq 24(%0), %%mm1             \n\t"\

            "paddw %2, %%mm0                \n\t"\

            "paddw %2, %%mm1                \n\t"\

            "psraw $5, %%mm0                \n\t"\

            "psraw $5, %%mm1                \n\t"\

            "packuswb %%mm1, %%mm0          \n\t"\

            OP_3DNOW(%%mm0, 8(%1), %%mm1, q)\

            :: "r"(temp), "r"(dst), "m"(ROUNDER)\

            : "memory"\

        );\

        dst+=dstStride;\

        src+=srcStride;\

    }\

}\

\

static void OPNAME ## mpeg4_qpel8_h_lowpass_mmx2(uint8_t *dst, uint8_t *src, int dstStride, int srcStride, int h){\

    asm volatile(\

        "pxor %%mm7, %%mm7                \n\t"\

        "1:                               \n\t"\

        "movq  (%0), %%mm0                \n\t" /* ABCDEFGH */\

        "movq %%mm0, %%mm1                \n\t" /* ABCDEFGH */\

        "movq %%mm0, %%mm2                \n\t" /* ABCDEFGH */\

        "punpcklbw %%mm7, %%mm0           \n\t" /* 0A0B0C0D */\

        "punpckhbw %%mm7, %%mm1           \n\t" /* 0E0F0G0H */\

        "pshufw $0x90, %%mm0, %%mm5       \n\t" /* 0A0A0B0C */\

        "pshufw $0x41, %%mm0, %%mm6       \n\t" /* 0B0A0A0B */\

        "movq %%mm2, %%mm3                \n\t" /* ABCDEFGH */\

        "movq %%mm2, %%mm4                \n\t" /* ABCDEFGH */\

        "psllq $8, %%mm2                  \n\t" /* 0ABCDEFG */\

        "psllq $16, %%mm3                 \n\t" /* 00ABCDEF */\

        "psllq $24, %%mm4                 \n\t" /* 000ABCDE */\

        "punpckhbw %%mm7, %%mm2           \n\t" /* 0D0E0F0G */\

        "punpckhbw %%mm7, %%mm3           \n\t" /* 0C0D0E0F */\

        "punpckhbw %%mm7, %%mm4           \n\t" /* 0B0C0D0E */\

        "paddw %%mm3, %%mm5               \n\t" /* b */\

        "paddw %%mm2, %%mm6               \n\t" /* c */\

        "paddw %%mm5, %%mm5               \n\t" /* 2b */\

        "psubw %%mm5, %%mm6               \n\t" /* c - 2b */\

        "pshufw $0x06, %%mm0, %%mm5       \n\t" /* 0C0B0A0A */\

        "pmullw "MANGLE(ff_pw_3)", %%mm6  \n\t" /* 3c - 6b */\

        "paddw %%mm4, %%mm0               \n\t" /* a */\

        "paddw %%mm1, %%mm5               \n\t" /* d */\

        "pmullw "MANGLE(ff_pw_20)", %%mm0 \n\t" /* 20a */\

        "psubw %%mm5, %%mm0               \n\t" /* 20a - d */\

        "paddw %5, %%mm6                  \n\t"\

        "paddw %%mm6, %%mm0               \n\t" /* 20a - 6b + 3c - d */\

        "psraw $5, %%mm0                  \n\t"\

        /* mm1=EFGH, mm2=DEFG, mm3=CDEF, mm4=BCDE, mm7=0 */\

        \

        "movd 5(%0), %%mm5                \n\t" /* FGHI */\

        "punpcklbw %%mm7, %%mm5           \n\t" /* 0F0G0H0I */\

        "pshufw $0xF9, %%mm5, %%mm6       \n\t" /* 0G0H0I0I */\

        "paddw %%mm5, %%mm1               \n\t" /* a */\

        "paddw %%mm6, %%mm2               \n\t" /* b */\

        "pshufw $0xBE, %%mm5, %%mm6       \n\t" /* 0H0I0I0H */\

        "pshufw $0x6F, %%mm5, %%mm5       \n\t" /* 0I0I0H0G */\

        "paddw %%mm6, %%mm3               \n\t" /* c */\

        "paddw %%mm5, %%mm4               \n\t" /* d */\

        "paddw %%mm2, %%mm2               \n\t" /* 2b */\

        "psubw %%mm2, %%mm3               \n\t" /* c - 2b */\

        "pmullw "MANGLE(ff_pw_20)", %%mm1 \n\t" /* 20a */\

        "pmullw "MANGLE(ff_pw_3)", %%mm3  \n\t" /* 3c - 6b */\

        "psubw %%mm4, %%mm3               \n\t" /* -6b + 3c - d */\

        "paddw %5, %%mm1                  \n\t"\

        "paddw %%mm1, %%mm3               \n\t" /* 20a - 6b + 3c - d */\

        "psraw $5, %%mm3                  \n\t"\

        "packuswb %%mm3, %%mm0            \n\t"\

        OP_MMX2(%%mm0, (%1), %%mm4, q)\

        \

        "add %3, %0                       \n\t"\

        "add %4, %1                       \n\t"\

        "decl %2                          \n\t"\

        " jnz 1b                          \n\t"\

        : "+a"(src), "+c"(dst), "+g"(h)\

        : "S"((long)srcStride), "D"((long)dstStride), /*"m"(ff_pw_20), "m"(ff_pw_3),*/ "m"(ROUNDER)\

        : "memory"\

    );\

}\

\

static void OPNAME ## mpeg4_qpel8_h_lowpass_3dnow(uint8_t *dst, uint8_t *src, int dstStride, int srcStride, int h){\

    int i;\

    int16_t temp[8];\

    /* quick HACK, XXX FIXME MUST be optimized */\

    for(i=0; i<h; i++)\

    {\

        temp[ 0]= (src[ 0]+src[ 1])*20 - (src[ 0]+src[ 2])*6 + (src[ 1]+src[ 3])*3 - (src[ 2]+src[ 4]);\

        temp[ 1]= (src[ 1]+src[ 2])*20 - (src[ 0]+src[ 3])*6 + (src[ 0]+src[ 4])*3 - (src[ 1]+src[ 5]);\

        temp[ 2]= (src[ 2]+src[ 3])*20 - (src[ 1]+src[ 4])*6 + (src[ 0]+src[ 5])*3 - (src[ 0]+src[ 6]);\

        temp[ 3]= (src[ 3]+src[ 4])*20 - (src[ 2]+src[ 5])*6 + (src[ 1]+src[ 6])*3 - (src[ 0]+src[ 7]);\

        temp[ 4]= (src[ 4]+src[ 5])*20 - (src[ 3]+src[ 6])*6 + (src[ 2]+src[ 7])*3 - (src[ 1]+src[ 8]);\

        temp[ 5]= (src[ 5]+src[ 6])*20 - (src[ 4]+src[ 7])*6 + (src[ 3]+src[ 8])*3 - (src[ 2]+src[ 8]);\

        temp[ 6]= (src[ 6]+src[ 7])*20 - (src[ 5]+src[ 8])*6 + (src[ 4]+src[ 8])*3 - (src[ 3]+src[ 7]);\

        temp[ 7]= (src[ 7]+src[ 8])*20 - (src[ 6]+src[ 8])*6 + (src[ 5]+src[ 7])*3 - (src[ 4]+src[ 6]);\

        asm volatile(\

            "movq (%0), %%mm0           \n\t"\

            "movq 8(%0), %%mm1          \n\t"\

            "paddw %2, %%mm0            \n\t"\

            "paddw %2, %%mm1            \n\t"\

            "psraw $5, %%mm0            \n\t"\

            "psraw $5, %%mm1            \n\t"\

            "packuswb %%mm1, %%mm0      \n\t"\

            OP_3DNOW(%%mm0, (%1), %%mm1, q)\

            :: "r"(temp), "r"(dst), "m"(ROUNDER)\

            :"memory"\

        );\

        dst+=dstStride;\

        src+=srcStride;\

    }\

}



#define QPEL_OP(OPNAME, ROUNDER, RND, OP, MMX)\

\

static void OPNAME ## mpeg4_qpel16_v_lowpass_ ## MMX(uint8_t *dst, uint8_t *src, int dstStride, int srcStride){\

    uint64_t temp[17*4];\

    uint64_t *temp_ptr= temp;\

    int count= 17;\

\

    /*FIXME unroll */\

    asm volatile(\

        "pxor %%mm7, %%mm7              \n\t"\

        "1:                             \n\t"\

        "movq (%0), %%mm0               \n\t"\

        "movq (%0), %%mm1               \n\t"\

        "movq 8(%0), %%mm2              \n\t"\

        "movq 8(%0), %%mm3              \n\t"\

        "punpcklbw %%mm7, %%mm0         \n\t"\

        "punpckhbw %%mm7, %%mm1         \n\t"\

        "punpcklbw %%mm7, %%mm2         \n\t"\

        "punpckhbw %%mm7, %%mm3         \n\t"\

        "movq %%mm0, (%1)               \n\t"\

        "movq %%mm1, 17*8(%1)           \n\t"\

        "movq %%mm2, 2*17*8(%1)         \n\t"\

        "movq %%mm3, 3*17*8(%1)         \n\t"\

        "add $8, %1                     \n\t"\

        "add %3, %0                     \n\t"\

        "decl %2                        \n\t"\

        " jnz 1b                        \n\t"\

        : "+r" (src), "+r" (temp_ptr), "+r"(count)\

        : "r" ((long)srcStride)\

        : "memory"\

    );\

    \

    temp_ptr= temp;\

    count=4;\

    \

/*FIXME reorder for speed */\

    asm volatile(\

        /*"pxor %%mm7, %%mm7              \n\t"*/\

        "1:                             \n\t"\

        "movq (%0), %%mm0               \n\t"\

        "movq 8(%0), %%mm1              \n\t"\

        "movq 16(%0), %%mm2             \n\t"\

        "movq 24(%0), %%mm3             \n\t"\

        QPEL_V_LOW(%%mm0, %%mm1, %%mm2, %%mm3, %5, %6, %5, 16(%0),  8(%0),   (%0), 32(%0), (%1), OP)\

        QPEL_V_LOW(%%mm1, %%mm2, %%mm3, %%mm0, %5, %6, %5,  8(%0),   (%0),   (%0), 40(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm2, %%mm3, %%mm0, %%mm1, %5, %6, %5,   (%0),   (%0),  8(%0), 48(%0), (%1), OP)\

        \

        QPEL_V_LOW(%%mm3, %%mm0, %%mm1, %%mm2, %5, %6, %5,   (%0),  8(%0), 16(%0), 56(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm0, %%mm1, %%mm2, %%mm3, %5, %6, %5,  8(%0), 16(%0), 24(%0), 64(%0), (%1), OP)\

        QPEL_V_LOW(%%mm1, %%mm2, %%mm3, %%mm0, %5, %6, %5, 16(%0), 24(%0), 32(%0), 72(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm2, %%mm3, %%mm0, %%mm1, %5, %6, %5, 24(%0), 32(%0), 40(%0), 80(%0), (%1), OP)\

        QPEL_V_LOW(%%mm3, %%mm0, %%mm1, %%mm2, %5, %6, %5, 32(%0), 40(%0), 48(%0), 88(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm0, %%mm1, %%mm2, %%mm3, %5, %6, %5, 40(%0), 48(%0), 56(%0), 96(%0), (%1), OP)\

        QPEL_V_LOW(%%mm1, %%mm2, %%mm3, %%mm0, %5, %6, %5, 48(%0), 56(%0), 64(%0),104(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm2, %%mm3, %%mm0, %%mm1, %5, %6, %5, 56(%0), 64(%0), 72(%0),112(%0), (%1), OP)\

        QPEL_V_LOW(%%mm3, %%mm0, %%mm1, %%mm2, %5, %6, %5, 64(%0), 72(%0), 80(%0),120(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm0, %%mm1, %%mm2, %%mm3, %5, %6, %5, 72(%0), 80(%0), 88(%0),128(%0), (%1), OP)\

        \

        QPEL_V_LOW(%%mm1, %%mm2, %%mm3, %%mm0, %5, %6, %5, 80(%0), 88(%0), 96(%0),128(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"  \

        QPEL_V_LOW(%%mm2, %%mm3, %%mm0, %%mm1, %5, %6, %5, 88(%0), 96(%0),104(%0),120(%0), (%1), OP)\

        QPEL_V_LOW(%%mm3, %%mm0, %%mm1, %%mm2, %5, %6, %5, 96(%0),104(%0),112(%0),112(%0), (%1, %3), OP)\

        \

        "add $136, %0                   \n\t"\

        "add %6, %1                     \n\t"\

        "decl %2                        \n\t"\

        " jnz 1b                        \n\t"\

        \

        : "+r"(temp_ptr), "+r"(dst), "+g"(count)\

        : "r"((long)dstStride), "r"(2*(long)dstStride), /*"m"(ff_pw_20), "m"(ff_pw_3),*/ "m"(ROUNDER), "g"(4-14*(long)dstStride)\

        :"memory"\

    );\

}\

\

static void OPNAME ## mpeg4_qpel8_v_lowpass_ ## MMX(uint8_t *dst, uint8_t *src, int dstStride, int srcStride){\

    uint64_t temp[9*2];\

    uint64_t *temp_ptr= temp;\

    int count= 9;\

\

    /*FIXME unroll */\

    asm volatile(\

        "pxor %%mm7, %%mm7              \n\t"\

        "1:                             \n\t"\

        "movq (%0), %%mm0               \n\t"\

        "movq (%0), %%mm1               \n\t"\

        "punpcklbw %%mm7, %%mm0         \n\t"\

        "punpckhbw %%mm7, %%mm1         \n\t"\

        "movq %%mm0, (%1)               \n\t"\

        "movq %%mm1, 9*8(%1)            \n\t"\

        "add $8, %1                     \n\t"\

        "add %3, %0                     \n\t"\

        "decl %2                        \n\t"\

        " jnz 1b                        \n\t"\

        : "+r" (src), "+r" (temp_ptr), "+r"(count)\

        : "r" ((long)srcStride)\

        : "memory"\

    );\

    \

    temp_ptr= temp;\

    count=2;\

    \

/*FIXME reorder for speed */\

    asm volatile(\

        /*"pxor %%mm7, %%mm7              \n\t"*/\

        "1:                             \n\t"\

        "movq (%0), %%mm0               \n\t"\

        "movq 8(%0), %%mm1              \n\t"\

        "movq 16(%0), %%mm2             \n\t"\

        "movq 24(%0), %%mm3             \n\t"\

        QPEL_V_LOW(%%mm0, %%mm1, %%mm2, %%mm3, %5, %6, %5, 16(%0),  8(%0),   (%0), 32(%0), (%1), OP)\

        QPEL_V_LOW(%%mm1, %%mm2, %%mm3, %%mm0, %5, %6, %5,  8(%0),   (%0),   (%0), 40(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm2, %%mm3, %%mm0, %%mm1, %5, %6, %5,   (%0),   (%0),  8(%0), 48(%0), (%1), OP)\

        \

        QPEL_V_LOW(%%mm3, %%mm0, %%mm1, %%mm2, %5, %6, %5,   (%0),  8(%0), 16(%0), 56(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm0, %%mm1, %%mm2, %%mm3, %5, %6, %5,  8(%0), 16(%0), 24(%0), 64(%0), (%1), OP)\

        \

        QPEL_V_LOW(%%mm1, %%mm2, %%mm3, %%mm0, %5, %6, %5, 16(%0), 24(%0), 32(%0), 64(%0), (%1, %3), OP)\

        "add %4, %1                     \n\t"\

        QPEL_V_LOW(%%mm2, %%mm3, %%mm0, %%mm1, %5, %6, %5, 24(%0), 32(%0), 40(%0), 56(%0), (%1), OP)\

        QPEL_V_LOW(%%mm3, %%mm0, %%mm1, %%mm2, %5, %6, %5, 32(%0), 40(%0), 48(%0), 48(%0), (%1, %3), OP)\

                \

        "add $72, %0                    \n\t"\

        "add %6, %1                     \n\t"\

        "decl %2                        \n\t"\

        " jnz 1b                        \n\t"\

         \

        : "+r"(temp_ptr), "+r"(dst), "+g"(count)\

        : "r"((long)dstStride), "r"(2*(long)dstStride), /*"m"(ff_pw_20), "m"(ff_pw_3),*/ "m"(ROUNDER), "g"(4-6*(long)dstStride)\

        : "memory"\

   );\

}\

\

static void OPNAME ## qpel8_mc00_ ## MMX (uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## pixels8_ ## MMX(dst, src, stride, 8);\

}\

\

static void OPNAME ## qpel8_mc10_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[8];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(half, src, 8, stride, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, src, half, stride, stride, 8);\

}\

\

static void OPNAME ## qpel8_mc20_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## mpeg4_qpel8_h_lowpass_ ## MMX(dst, src, stride, stride, 8);\

}\

\

static void OPNAME ## qpel8_mc30_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[8];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(half, src, 8, stride, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, src+1, half, stride, stride, 8);\

}\

\

static void OPNAME ## qpel8_mc01_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[8];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(half, src, 8, stride);\

    OPNAME ## pixels8_l2_ ## MMX(dst, src, half, stride, stride, 8);\

}\

\

static void OPNAME ## qpel8_mc02_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## mpeg4_qpel8_v_lowpass_ ## MMX(dst, src, stride, stride);\

}\

\

static void OPNAME ## qpel8_mc03_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[8];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(half, src, 8, stride);\

    OPNAME ## pixels8_l2_ ## MMX(dst, src+stride, half, stride, stride, 8);\

}\

static void OPNAME ## qpel8_mc11_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half) + 64;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## pixels8_l2_ ## MMX(halfH, src, halfH, 8, stride, 9);\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(halfHV, halfH, 8, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, halfH, halfHV, stride, 8, 8);\

}\

static void OPNAME ## qpel8_mc31_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half) + 64;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## pixels8_l2_ ## MMX(halfH, src+1, halfH, 8, stride, 9);\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(halfHV, halfH, 8, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, halfH, halfHV, stride, 8, 8);\

}\

static void OPNAME ## qpel8_mc13_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half) + 64;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## pixels8_l2_ ## MMX(halfH, src, halfH, 8, stride, 9);\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(halfHV, halfH, 8, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, halfH+8, halfHV, stride, 8, 8);\

}\

static void OPNAME ## qpel8_mc33_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half) + 64;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## pixels8_l2_ ## MMX(halfH, src+1, halfH, 8, stride, 9);\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(halfHV, halfH, 8, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, halfH+8, halfHV, stride, 8, 8);\

}\

static void OPNAME ## qpel8_mc21_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half) + 64;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(halfHV, halfH, 8, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, halfH, halfHV, stride, 8, 8);\

}\

static void OPNAME ## qpel8_mc23_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half) + 64;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## mpeg4_qpel8_v_lowpass_ ## MMX(halfHV, halfH, 8, 8);\

    OPNAME ## pixels8_l2_ ## MMX(dst, halfH+8, halfHV, stride, 8, 8);\

}\

static void OPNAME ## qpel8_mc12_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## pixels8_l2_ ## MMX(halfH, src, halfH, 8, stride, 9);\

    OPNAME ## mpeg4_qpel8_v_lowpass_ ## MMX(dst, halfH, stride, 8);\

}\

static void OPNAME ## qpel8_mc32_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[8 + 9];\

    uint8_t * const halfH= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    put ## RND ## pixels8_l2_ ## MMX(halfH, src+1, halfH, 8, stride, 9);\

    OPNAME ## mpeg4_qpel8_v_lowpass_ ## MMX(dst, halfH, stride, 8);\

}\

static void OPNAME ## qpel8_mc22_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[9];\

    uint8_t * const halfH= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel8_h_lowpass_ ## MMX(halfH, src, 8, stride, 9);\

    OPNAME ## mpeg4_qpel8_v_lowpass_ ## MMX(dst, halfH, stride, 8);\

}\

static void OPNAME ## qpel16_mc00_ ## MMX (uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## pixels16_ ## MMX(dst, src, stride, 16);\

}\

\

static void OPNAME ## qpel16_mc10_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[32];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(half, src, 16, stride, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, src, half, stride, stride, 16);\

}\

\

static void OPNAME ## qpel16_mc20_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## mpeg4_qpel16_h_lowpass_ ## MMX(dst, src, stride, stride, 16);\

}\

\

static void OPNAME ## qpel16_mc30_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[32];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(half, src, 16, stride, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, src+1, half, stride, stride, 16);\

}\

\

static void OPNAME ## qpel16_mc01_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[32];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(half, src, 16, stride);\

    OPNAME ## pixels16_l2_ ## MMX(dst, src, half, stride, stride, 16);\

}\

\

static void OPNAME ## qpel16_mc02_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## mpeg4_qpel16_v_lowpass_ ## MMX(dst, src, stride, stride);\

}\

\

static void OPNAME ## qpel16_mc03_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t temp[32];\

    uint8_t * const half= (uint8_t*)temp;\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(half, src, 16, stride);\

    OPNAME ## pixels16_l2_ ## MMX(dst, src+stride, half, stride, stride, 16);\

}\

static void OPNAME ## qpel16_mc11_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[16*2 + 17*2];\

    uint8_t * const halfH= ((uint8_t*)half) + 256;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## pixels16_l2_ ## MMX(halfH, src, halfH, 16, stride, 17);\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(halfHV, halfH, 16, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, halfH, halfHV, stride, 16, 16);\

}\

static void OPNAME ## qpel16_mc31_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[16*2 + 17*2];\

    uint8_t * const halfH= ((uint8_t*)half) + 256;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## pixels16_l2_ ## MMX(halfH, src+1, halfH, 16, stride, 17);\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(halfHV, halfH, 16, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, halfH, halfHV, stride, 16, 16);\

}\

static void OPNAME ## qpel16_mc13_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[16*2 + 17*2];\

    uint8_t * const halfH= ((uint8_t*)half) + 256;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## pixels16_l2_ ## MMX(halfH, src, halfH, 16, stride, 17);\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(halfHV, halfH, 16, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, halfH+16, halfHV, stride, 16, 16);\

}\

static void OPNAME ## qpel16_mc33_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[16*2 + 17*2];\

    uint8_t * const halfH= ((uint8_t*)half) + 256;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## pixels16_l2_ ## MMX(halfH, src+1, halfH, 16, stride, 17);\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(halfHV, halfH, 16, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, halfH+16, halfHV, stride, 16, 16);\

}\

static void OPNAME ## qpel16_mc21_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[16*2 + 17*2];\

    uint8_t * const halfH= ((uint8_t*)half) + 256;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(halfHV, halfH, 16, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, halfH, halfHV, stride, 16, 16);\

}\

static void OPNAME ## qpel16_mc23_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[16*2 + 17*2];\

    uint8_t * const halfH= ((uint8_t*)half) + 256;\

    uint8_t * const halfHV= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## mpeg4_qpel16_v_lowpass_ ## MMX(halfHV, halfH, 16, 16);\

    OPNAME ## pixels16_l2_ ## MMX(dst, halfH+16, halfHV, stride, 16, 16);\

}\

static void OPNAME ## qpel16_mc12_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[17*2];\

    uint8_t * const halfH= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## pixels16_l2_ ## MMX(halfH, src, halfH, 16, stride, 17);\

    OPNAME ## mpeg4_qpel16_v_lowpass_ ## MMX(dst, halfH, stride, 16);\

}\

static void OPNAME ## qpel16_mc32_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[17*2];\

    uint8_t * const halfH= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    put ## RND ## pixels16_l2_ ## MMX(halfH, src+1, halfH, 16, stride, 17);\

    OPNAME ## mpeg4_qpel16_v_lowpass_ ## MMX(dst, halfH, stride, 16);\

}\

static void OPNAME ## qpel16_mc22_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    uint64_t half[17*2];\

    uint8_t * const halfH= ((uint8_t*)half);\

    put ## RND ## mpeg4_qpel16_h_lowpass_ ## MMX(halfH, src, 16, stride, 17);\

    OPNAME ## mpeg4_qpel16_v_lowpass_ ## MMX(dst, halfH, stride, 16);\

}



#define PUT_OP(a,b,temp, size) "mov" #size " " #a ", " #b "        \n\t"

#define AVG_3DNOW_OP(a,b,temp, size) \

"mov" #size " " #b ", " #temp "   \n\t"\

"pavgusb " #temp ", " #a "        \n\t"\

"mov" #size " " #a ", " #b "      \n\t"

#define AVG_MMX2_OP(a,b,temp, size) \

"mov" #size " " #b ", " #temp "   \n\t"\

"pavgb " #temp ", " #a "          \n\t"\

"mov" #size " " #a ", " #b "      \n\t"



QPEL_BASE(put_       , ff_pw_16, _       , PUT_OP, PUT_OP)

QPEL_BASE(avg_       , ff_pw_16, _       , AVG_MMX2_OP, AVG_3DNOW_OP)

QPEL_BASE(put_no_rnd_, ff_pw_15, _no_rnd_, PUT_OP, PUT_OP)

QPEL_OP(put_       , ff_pw_16, _       , PUT_OP, 3dnow)

QPEL_OP(avg_       , ff_pw_16, _       , AVG_3DNOW_OP, 3dnow)

QPEL_OP(put_no_rnd_, ff_pw_15, _no_rnd_, PUT_OP, 3dnow)

QPEL_OP(put_       , ff_pw_16, _       , PUT_OP, mmx2)

QPEL_OP(avg_       , ff_pw_16, _       , AVG_MMX2_OP, mmx2)

QPEL_OP(put_no_rnd_, ff_pw_15, _no_rnd_, PUT_OP, mmx2)



/***********************************/

/* bilinear qpel: not compliant to any spec, only for -lavdopts fast */



#define QPEL_2TAP_XY(OPNAME, SIZE, MMX, XY, HPEL)\

static void OPNAME ## 2tap_qpel ## SIZE ## _mc ## XY ## _ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## pixels ## SIZE ## HPEL(dst, src, stride, SIZE);\

}

#define QPEL_2TAP_L3(OPNAME, SIZE, MMX, XY, S0, S1, S2)\

static void OPNAME ## 2tap_qpel ## SIZE ## _mc ## XY ## _ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## 2tap_qpel ## SIZE ## _l3_ ## MMX(dst, src+S0, stride, SIZE, S1, S2);\

}



#define QPEL_2TAP(OPNAME, SIZE, MMX)\

QPEL_2TAP_XY(OPNAME, SIZE, MMX, 20, _x2_ ## MMX)\

QPEL_2TAP_XY(OPNAME, SIZE, MMX, 02, _y2_ ## MMX)\

QPEL_2TAP_XY(OPNAME, SIZE, MMX, 22, _xy2_mmx)\

static const qpel_mc_func OPNAME ## 2tap_qpel ## SIZE ## _mc00_ ## MMX =\

                          OPNAME ## qpel ## SIZE ## _mc00_ ## MMX;\

static const qpel_mc_func OPNAME ## 2tap_qpel ## SIZE ## _mc21_ ## MMX =\

                          OPNAME ## 2tap_qpel ## SIZE ## _mc20_ ## MMX;\

static const qpel_mc_func OPNAME ## 2tap_qpel ## SIZE ## _mc12_ ## MMX =\

                          OPNAME ## 2tap_qpel ## SIZE ## _mc02_ ## MMX;\

static void OPNAME ## 2tap_qpel ## SIZE ## _mc32_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## pixels ## SIZE ## _y2_ ## MMX(dst, src+1, stride, SIZE);\

}\

static void OPNAME ## 2tap_qpel ## SIZE ## _mc23_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    OPNAME ## pixels ## SIZE ## _x2_ ## MMX(dst, src+stride, stride, SIZE);\

}\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 10, 0,         1,       0)\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 30, 1,        -1,       0)\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 01, 0,         stride,  0)\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 03, stride,   -stride,  0)\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 11, 0,         stride,  1)\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 31, 1,         stride, -1)\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 13, stride,   -stride,  1)\

QPEL_2TAP_L3(OPNAME, SIZE, MMX, 33, stride+1, -stride, -1)\



QPEL_2TAP(put_, 16, mmx2)

QPEL_2TAP(avg_, 16, mmx2)

QPEL_2TAP(put_,  8, mmx2)

QPEL_2TAP(avg_,  8, mmx2)

QPEL_2TAP(put_, 16, 3dnow)

QPEL_2TAP(avg_, 16, 3dnow)

QPEL_2TAP(put_,  8, 3dnow)

QPEL_2TAP(avg_,  8, 3dnow)





#if 0

static void just_return() { return; }

#endif



static void gmc_mmx(uint8_t *dst, uint8_t *src, int stride, int h, int ox, int oy,

                    int dxx, int dxy, int dyx, int dyy, int shift, int r, int width, int height){

    const int w = 8;

    const int ix = ox>>(16+shift);

    const int iy = oy>>(16+shift);

    const int oxs = ox>>4;

    const int oys = oy>>4;

    const int dxxs = dxx>>4;

    const int dxys = dxy>>4;

    const int dyxs = dyx>>4;

    const int dyys = dyy>>4;

    const uint16_t r4[4] = {r,r,r,r};

    const uint16_t dxy4[4] = {dxys,dxys,dxys,dxys};

    const uint16_t dyy4[4] = {dyys,dyys,dyys,dyys};

    const uint64_t shift2 = 2*shift;

    uint8_t edge_buf[(h+1)*stride];

    int x, y;



    const int dxw = (dxx-(1<<(16+shift)))*(w-1);

    const int dyh = (dyy-(1<<(16+shift)))*(h-1);

    const int dxh = dxy*(h-1);

    const int dyw = dyx*(w-1);

    if( // non-constant fullpel offset (3% of blocks)

        ((ox^(ox+dxw)) | (ox^(ox+dxh)) | (ox^(ox+dxw+dxh)) |

         (oy^(oy+dyw)) | (oy^(oy+dyh)) | (oy^(oy+dyw+dyh))) >> (16+shift)

        // uses more than 16 bits of subpel mv (only at huge resolution)

        || (dxx|dxy|dyx|dyy)&15 )

    {

        //FIXME could still use mmx for some of the rows

        ff_gmc_c(dst, src, stride, h, ox, oy, dxx, dxy, dyx, dyy, shift, r, width, height);

        return;

    }



    src += ix + iy*stride;

    if( (unsigned)ix >= width-w ||

        (unsigned)iy >= height-h )

    {

        ff_emulated_edge_mc(edge_buf, src, stride, w+1, h+1, ix, iy, width, height);

        src = edge_buf;

    }



    asm volatile(

        "movd         %0, %%mm6 \n\t"

        "pxor      %%mm7, %%mm7 \n\t"

        "punpcklwd %%mm6, %%mm6 \n\t"

        "punpcklwd %%mm6, %%mm6 \n\t"

        :: "r"(1<<shift)

    );



    for(x=0; x<w; x+=4){

        uint16_t dx4[4] = { oxs - dxys + dxxs*(x+0),

                            oxs - dxys + dxxs*(x+1),

                            oxs - dxys + dxxs*(x+2),

                            oxs - dxys + dxxs*(x+3) };

        uint16_t dy4[4] = { oys - dyys + dyxs*(x+0),

                            oys - dyys + dyxs*(x+1),

                            oys - dyys + dyxs*(x+2),

                            oys - dyys + dyxs*(x+3) };



        for(y=0; y<h; y++){

            asm volatile(

                "movq   %0,  %%mm4 \n\t"

                "movq   %1,  %%mm5 \n\t"

                "paddw  %2,  %%mm4 \n\t"

                "paddw  %3,  %%mm5 \n\t"

                "movq   %%mm4, %0  \n\t"

                "movq   %%mm5, %1  \n\t"

                "psrlw  $12, %%mm4 \n\t"

                "psrlw  $12, %%mm5 \n\t"

                : "+m"(*dx4), "+m"(*dy4)

                : "m"(*dxy4), "m"(*dyy4)

            );



            asm volatile(

                "movq   %%mm6, %%mm2 \n\t"

                "movq   %%mm6, %%mm1 \n\t"

                "psubw  %%mm4, %%mm2 \n\t"

                "psubw  %%mm5, %%mm1 \n\t"

                "movq   %%mm2, %%mm0 \n\t"

                "movq   %%mm4, %%mm3 \n\t"

                "pmullw %%mm1, %%mm0 \n\t" // (s-dx)*(s-dy)

                "pmullw %%mm5, %%mm3 \n\t" // dx*dy

                "pmullw %%mm5, %%mm2 \n\t" // (s-dx)*dy

                "pmullw %%mm4, %%mm1 \n\t" // dx*(s-dy)



                "movd   %4,    %%mm5 \n\t"

                "movd   %3,    %%mm4 \n\t"

                "punpcklbw %%mm7, %%mm5 \n\t"

                "punpcklbw %%mm7, %%mm4 \n\t"

                "pmullw %%mm5, %%mm3 \n\t" // src[1,1] * dx*dy

                "pmullw %%mm4, %%mm2 \n\t" // src[0,1] * (s-dx)*dy



                "movd   %2,    %%mm5 \n\t"

                "movd   %1,    %%mm4 \n\t"

                "punpcklbw %%mm7, %%mm5 \n\t"

                "punpcklbw %%mm7, %%mm4 \n\t"

                "pmullw %%mm5, %%mm1 \n\t" // src[1,0] * dx*(s-dy)

                "pmullw %%mm4, %%mm0 \n\t" // src[0,0] * (s-dx)*(s-dy)

                "paddw  %5,    %%mm1 \n\t"

                "paddw  %%mm3, %%mm2 \n\t"

                "paddw  %%mm1, %%mm0 \n\t"

                "paddw  %%mm2, %%mm0 \n\t"



                "psrlw    %6,    %%mm0 \n\t"

                "packuswb %%mm0, %%mm0 \n\t"

                "movd     %%mm0, %0    \n\t"



                : "=m"(dst[x+y*stride])

                : "m"(src[0]), "m"(src[1]),

                  "m"(src[stride]), "m"(src[stride+1]),

                  "m"(*r4), "m"(shift2)

            );

            src += stride;

        }

        src += 4-h*stride;

    }

}
