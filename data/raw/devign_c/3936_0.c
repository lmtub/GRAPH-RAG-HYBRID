static inline void assert_fp_access_checked(DisasContext *s)

{

#ifdef CONFIG_DEBUG_TCG

    if (unlikely(!s->fp_access_checked || !s->cpacr_fpen)) {

        fprintf(stderr, "target-arm: FP access check missing for "

                "instruction 0x%08x\n", s->insn);

        abort();

    }

#endif

}
