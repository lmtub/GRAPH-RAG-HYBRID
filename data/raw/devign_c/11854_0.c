void helper_mtc0_entryhi(CPUMIPSState *env, target_ulong arg1)

{

    target_ulong old, val, mask;

    mask = (TARGET_PAGE_MASK << 1) | env->CP0_EntryHi_ASID_mask;

    if (((env->CP0_Config4 >> CP0C4_IE) & 0x3) >= 2) {

        mask |= 1 << CP0EnHi_EHINV;

    }



    /* 1k pages not implemented */

#if defined(TARGET_MIPS64)

    if (env->insn_flags & ISA_MIPS32R6) {

        int entryhi_r = extract64(arg1, 62, 2);

        int config0_at = extract32(env->CP0_Config0, 13, 2);

        bool no_supervisor = (env->CP0_Status_rw_bitmask & 0x8) == 0;

        if ((entryhi_r == 2) ||

            (entryhi_r == 1 && (no_supervisor || config0_at == 1))) {

            /* skip EntryHi.R field if new value is reserved */

            mask &= ~(0x3ull << 62);

        }

    }

    mask &= env->SEGMask;

#endif

    old = env->CP0_EntryHi;

    val = (arg1 & mask) | (old & ~mask);

    env->CP0_EntryHi = val;

    if (env->CP0_Config3 & (1 << CP0C3_MT)) {

        sync_c0_entryhi(env, env->current_tc);

    }

    /* If the ASID changes, flush qemu's TLB.  */

    if ((old & env->CP0_EntryHi_ASID_mask) !=

        (val & env->CP0_EntryHi_ASID_mask)) {

        cpu_mips_tlb_flush(env);

    }

}
