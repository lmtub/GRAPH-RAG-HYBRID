static bool x86_cpu_has_work(CPUState *cs)

{

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;



    return ((cs->interrupt_request & (CPU_INTERRUPT_HARD |

                                      CPU_INTERRUPT_POLL)) &&

            (env->eflags & IF_MASK)) ||

           (cs->interrupt_request & (CPU_INTERRUPT_NMI |

                                     CPU_INTERRUPT_INIT |

                                     CPU_INTERRUPT_SIPI |

                                     CPU_INTERRUPT_MCE));

}
