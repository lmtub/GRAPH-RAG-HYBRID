static inline void tcg_out_ld_ptr(TCGContext *s, int ret,

                                  tcg_target_long arg)

{

#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

    if (arg != (arg & 0xffffffff))

        fprintf(stderr, "unimplemented %s with offset %ld\n", __func__, arg);

    if (arg != (arg & 0xfff))

        tcg_out32(s, SETHI | INSN_RD(ret) | (((uint32_t)arg & 0xfffffc00) >> 10));

    tcg_out32(s, LDX | INSN_RD(ret) | INSN_RS1(ret) |

              INSN_IMM13(arg & 0x3ff));

#else

    tcg_out_ld_raw(s, ret, arg);

#endif

}
