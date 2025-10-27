void helper_mtc0_pagemask(CPUMIPSState *env, target_ulong arg1)

{

    /* 1k pages not implemented */

    env->CP0_PageMask = arg1 & (0x1FFFFFFF & (TARGET_PAGE_MASK << 1));

}
