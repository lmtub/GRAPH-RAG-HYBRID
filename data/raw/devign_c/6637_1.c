static inline void check_hwrena(CPUMIPSState *env, int reg)

{

    if ((env->hflags & MIPS_HFLAG_CP0) || (env->CP0_HWREna & (1 << reg))) {

        return;

    }

    do_raise_exception(env, EXCP_RI, GETPC());

}
