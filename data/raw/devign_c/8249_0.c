void helper_sysexit(int dflag)

{

    int cpl;



    cpl = env->hflags & HF_CPL_MASK;

    if (env->sysenter_cs == 0 || cpl != 0) {

        raise_exception_err(EXCP0D_GPF, 0);

    }

    cpu_x86_set_cpl(env, 3);

#ifdef TARGET_X86_64

    if (dflag == 2) {

        cpu_x86_load_seg_cache(env, R_CS, ((env->sysenter_cs + 32) & 0xfffc) | 3,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK | DESC_L_MASK);

        cpu_x86_load_seg_cache(env, R_SS, ((env->sysenter_cs + 40) & 0xfffc) | 3,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_W_MASK | DESC_A_MASK);

    } else

#endif

    {

        cpu_x86_load_seg_cache(env, R_CS, ((env->sysenter_cs + 16) & 0xfffc) | 3,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK);

        cpu_x86_load_seg_cache(env, R_SS, ((env->sysenter_cs + 24) & 0xfffc) | 3,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_W_MASK | DESC_A_MASK);

    }

    ESP = ECX;

    EIP = EDX;

#ifdef CONFIG_KQEMU

    if (kqemu_is_ok(env)) {

        env->exception_index = -1;

        cpu_loop_exit();

    }

#endif

}
