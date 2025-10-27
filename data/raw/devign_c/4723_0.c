int kvm_arch_insert_sw_breakpoint(CPUState *env, struct kvm_sw_breakpoint *bp)

{

    static const uint8_t int3 = 0xcc;



    if (cpu_memory_rw_debug(env, bp->pc, (uint8_t *)&bp->saved_insn, 1, 0) ||

        cpu_memory_rw_debug(env, bp->pc, (uint8_t *)&int3, 1, 1))

        return -EINVAL;

    return 0;

}
