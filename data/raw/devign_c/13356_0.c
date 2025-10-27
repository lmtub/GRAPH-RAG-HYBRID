uint32_t HELPER(get_cp15)(CPUState *env, uint32_t insn)

{

    cpu_abort(env, "cp15 insn %08x\n", insn);

    return 0;

}
