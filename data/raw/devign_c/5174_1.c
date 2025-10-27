static bool do_modify_softint(CPUSPARCState *env, uint32_t value)

{

    if (env->softint != value) {

        env->softint = value;

#if !defined(CONFIG_USER_ONLY)

        if (cpu_interrupts_enabled(env)) {


            cpu_check_irqs(env);


        }

#endif

        return true;

    }

    return false;

}