static int get_cpsr(QEMUFile *f, void *opaque, size_t size)

{

    ARMCPU *cpu = opaque;

    CPUARMState *env = &cpu->env;

    uint32_t val = qemu_get_be32(f);



    env->aarch64 = ((val & PSTATE_nRW) == 0);



    if (is_a64(env)) {

        pstate_write(env, val);

        return 0;

    }



    /* Avoid mode switch when restoring CPSR */

    env->uncached_cpsr = val & CPSR_M;

    cpsr_write(env, val, 0xffffffff, CPSRWriteRaw);

    return 0;

}
