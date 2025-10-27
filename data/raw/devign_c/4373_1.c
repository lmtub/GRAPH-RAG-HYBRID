void helper_wrpstate(CPUSPARCState *env, target_ulong new_state)

{

    cpu_change_pstate(env, new_state & 0xf3f);



#if !defined(CONFIG_USER_ONLY)

    if (cpu_interrupts_enabled(env)) {


        cpu_check_irqs(env);


    }

#endif

}