static av_always_inline void h264_loop_filter_strength_iteration_mmx2(int16_t bS[2][4][4], uint8_t nnz[40],

                                                                      int8_t ref[2][40],   int16_t mv[2][40][2],

                                                                      int bidir,   int edges, int step,

                                                                      int mask_mv, int dir)

{

        const x86_reg d_idx = dir ? -8 : -1;

        DECLARE_ALIGNED(8, const uint64_t, mask_dir) = dir ? 0 : 0xffffffffffffffffULL;

        int b_idx, edge;

        for( b_idx=12, edge=0; edge<edges; edge+=step, b_idx+=8*step ) {

            __asm__ volatile(

                "pand %0, %%mm0 \n\t"

                ::"m"(mask_dir)

            );

            if(!(mask_mv & edge)) {

                if(bidir) {

                    __asm__ volatile(

                        "movd         (%1,%0), %%mm2 \n"

                        "punpckldq  40(%1,%0), %%mm2 \n" // { ref0[bn], ref1[bn] }

                        "pshufw $0x44,   (%1), %%mm0 \n" // { ref0[b], ref0[b] }

                        "pshufw $0x44, 40(%1), %%mm1 \n" // { ref1[b], ref1[b] }

                        "pshufw $0x4E, %%mm2, %%mm3 \n"

                        "psubb         %%mm2, %%mm0 \n" // { ref0[b]!=ref0[bn], ref0[b]!=ref1[bn] }

                        "psubb         %%mm3, %%mm1 \n" // { ref1[b]!=ref1[bn], ref1[b]!=ref0[bn] }

                        "1: \n"

                        "por           %%mm1, %%mm0 \n"

                        "movq      (%2,%0,4), %%mm1 \n"

                        "movq     8(%2,%0,4), %%mm2 \n"

                        "movq          %%mm1, %%mm3 \n"

                        "movq          %%mm2, %%mm4 \n"

                        "psubw          (%2), %%mm1 \n"

                        "psubw         8(%2), %%mm2 \n"

                        "psubw       160(%2), %%mm3 \n"

                        "psubw       168(%2), %%mm4 \n"

                        "packsswb      %%mm2, %%mm1 \n"

                        "packsswb      %%mm4, %%mm3 \n"

                        "paddb         %%mm6, %%mm1 \n"

                        "paddb         %%mm6, %%mm3 \n"

                        "psubusb       %%mm5, %%mm1 \n" // abs(mv[b] - mv[bn]) >= limit

                        "psubusb       %%mm5, %%mm3 \n"

                        "packsswb      %%mm3, %%mm1 \n"

                        "add $40, %0 \n"

                        "cmp $40, %0 \n"

                        "jl 1b \n"

                        "sub $80, %0 \n"

                        "pshufw $0x4E, %%mm1, %%mm1 \n"

                        "por           %%mm1, %%mm0 \n"

                        "pshufw $0x4E, %%mm0, %%mm1 \n"

                        "pminub        %%mm1, %%mm0 \n"

                        ::"r"(d_idx),

                          "r"(ref[0]+b_idx),

                          "r"(mv[0]+b_idx)

                    );

                } else {

                    __asm__ volatile(

                        "movd        (%1), %%mm0 \n"

                        "psubb    (%1,%0), %%mm0 \n" // ref[b] != ref[bn]

                        "movq        (%2), %%mm1 \n"

                        "movq       8(%2), %%mm2 \n"

                        "psubw  (%2,%0,4), %%mm1 \n"

                        "psubw 8(%2,%0,4), %%mm2 \n"

                        "packsswb   %%mm2, %%mm1 \n"

                        "paddb      %%mm6, %%mm1 \n"

                        "psubusb    %%mm5, %%mm1 \n" // abs(mv[b] - mv[bn]) >= limit

                        "packsswb   %%mm1, %%mm1 \n"

                        "por        %%mm1, %%mm0 \n"

                        ::"r"(d_idx),

                          "r"(ref[0]+b_idx),

                          "r"(mv[0]+b_idx)

                    );

                }

            }

            __asm__ volatile(

                "movd %0, %%mm1 \n"

                "por  %1, %%mm1 \n" // nnz[b] || nnz[bn]

                ::"m"(nnz[b_idx]),

                  "m"(nnz[b_idx+d_idx])

            );

            __asm__ volatile(

                "pminub    %%mm7, %%mm1 \n"

                "pminub    %%mm7, %%mm0 \n"

                "psllw        $1, %%mm1 \n"

                "pxor      %%mm2, %%mm2 \n"

                "pmaxub    %%mm0, %%mm1 \n"

                "punpcklbw %%mm2, %%mm1 \n"

                "movq      %%mm1, %0    \n"

                :"=m"(*bS[dir][edge])

                ::"memory"

            );

        }

}
