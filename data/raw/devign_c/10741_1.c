int kvm_arch_insert_sw_breakpoint(CPUState *cs, struct kvm_sw_breakpoint *bp)

{



    if (cpu_memory_rw_debug(cs, bp->pc, (uint8_t *)&bp->saved_insn,

                            sizeof(diag_501), 0) ||

        cpu_memory_rw_debug(cs, bp->pc, (uint8_t *)diag_501,

                            sizeof(diag_501), 1)) {

        return -EINVAL;

    }

    return 0;

}
