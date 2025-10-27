static void aarch64_cpu_set_pc(CPUState *cs, vaddr value)

{

    ARMCPU *cpu = ARM_CPU(cs);

    /*

     * TODO: this will need updating for system emulation,

     * when the core may be in AArch32 mode.

     */

    cpu->env.pc = value;

}
