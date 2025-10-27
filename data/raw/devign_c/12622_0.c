int kvm_arch_remove_sw_breakpoint(CPUState *env, struct kvm_sw_breakpoint *bp)

{

    uint8_t int3;



    if (cpu_memory_rw_debug(env, bp->pc, &int3, 1, 0) || int3 != 0xcc ||

        cpu_memory_rw_debug(env, bp->pc, (uint8_t *)&bp->saved_insn, 1, 1))

        return -EINVAL;

    return 0;

}
