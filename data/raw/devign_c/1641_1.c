static void v7m_exception_taken(ARMCPU *cpu, uint32_t lr)

{

    /* Do the "take the exception" parts of exception entry,

     * but not the pushing of state to the stack. This is

     * similar to the pseudocode ExceptionTaken() function.

     */

    CPUARMState *env = &cpu->env;

    uint32_t addr;



    armv7m_nvic_acknowledge_irq(env->nvic);

    write_v7m_control_spsel(env, 0);

    arm_clear_exclusive(env);

    /* Clear IT bits */

    env->condexec_bits = 0;

    env->regs[14] = lr;

    addr = arm_v7m_load_vector(cpu);

    env->regs[15] = addr & 0xfffffffe;

    env->thumb = addr & 1;

}
