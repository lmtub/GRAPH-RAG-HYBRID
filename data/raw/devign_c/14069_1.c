void cpu_check_irqs(CPUSPARCState *env)
{
    CPUState *cs;
    if (env->pil_in && (env->interrupt_index == 0 ||
                        (env->interrupt_index & ~15) == TT_EXTINT)) {
        unsigned int i;
        for (i = 15; i > 0; i--) {
            if (env->pil_in & (1 << i)) {
                int old_interrupt = env->interrupt_index;
                env->interrupt_index = TT_EXTINT | i;
                if (old_interrupt != env->interrupt_index) {
                    cs = CPU(sparc_env_get_cpu(env));
                    trace_sun4m_cpu_interrupt(i);
                    cpu_interrupt(cs, CPU_INTERRUPT_HARD);
                }
                break;
            }
        }
    } else if (!env->pil_in && (env->interrupt_index & ~15) == TT_EXTINT) {
        cs = CPU(sparc_env_get_cpu(env));
        trace_sun4m_cpu_reset_interrupt(env->interrupt_index & 15);
        env->interrupt_index = 0;
        cpu_reset_interrupt(cs, CPU_INTERRUPT_HARD);
    }
}