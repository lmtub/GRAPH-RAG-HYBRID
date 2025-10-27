static inline void tcg_out_addi(TCGContext *s, int reg, tcg_target_long val)

{

    if (val != 0) {

        if (val == (val & 0xfff))

            tcg_out_arithi(s, reg, reg, val, ARITH_ADD);

        else

            fprintf(stderr, "unimplemented addi %ld\n", (long)val);

    }

}
