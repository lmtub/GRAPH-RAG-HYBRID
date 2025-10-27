void helper_sysret(CPUX86State *env, int dflag)

{

    int cpl, selector;



    if (!(env->efer & MSR_EFER_SCE)) {

        raise_exception_err(env, EXCP06_ILLOP, 0);

    }

    cpl = env->hflags & HF_CPL_MASK;

    if (!(env->cr[0] & CR0_PE_MASK) || cpl != 0) {

        raise_exception_err(env, EXCP0D_GPF, 0);

    }

    selector = (env->star >> 48) & 0xffff;

    if (env->hflags & HF_LMA_MASK) {

        cpu_load_eflags(env, (uint32_t)(env->regs[11]), TF_MASK | AC_MASK

                        | ID_MASK | IF_MASK | IOPL_MASK | VM_MASK | RF_MASK |

                        NT_MASK);

        if (dflag == 2) {

            cpu_x86_load_seg_cache(env, R_CS, (selector + 16) | 3,

                                   0, 0xffffffff,

                                   DESC_G_MASK | DESC_P_MASK |

                                   DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                                   DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK |

                                   DESC_L_MASK);

            env->eip = env->regs[R_ECX];

        } else {

            cpu_x86_load_seg_cache(env, R_CS, selector | 3,

                                   0, 0xffffffff,

                                   DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                                   DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                                   DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK);

            env->eip = (uint32_t)env->regs[R_ECX];

        }

        cpu_x86_load_seg_cache(env, R_SS, selector + 8,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_W_MASK | DESC_A_MASK);

    } else {

        env->eflags |= IF_MASK;

        cpu_x86_load_seg_cache(env, R_CS, selector | 3,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK);

        env->eip = (uint32_t)env->regs[R_ECX];

        cpu_x86_load_seg_cache(env, R_SS, selector + 8,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_W_MASK | DESC_A_MASK);

    }

}
