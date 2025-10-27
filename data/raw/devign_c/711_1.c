int kvm_arch_remove_sw_breakpoint(CPUState *cs, struct kvm_sw_breakpoint *bp)

{

    uint8_t t[sizeof(diag_501)];



    if (cpu_memory_rw_debug(cs, bp->pc, t, sizeof(diag_501), 0)) {

        return -EINVAL;

    } else if (memcmp(t, diag_501, sizeof(diag_501))) {

        return -EINVAL;

    } else if (cpu_memory_rw_debug(cs, bp->pc, (uint8_t *)&bp->saved_insn,

                                   sizeof(diag_501), 1)) {

        return -EINVAL;

    }



    return 0;

}
