void helper_sysret(int dflag)

{

    int cpl, selector;



    if (!(env->efer & MSR_EFER_SCE)) {

        raise_exception_err(EXCP06_ILLOP, 0);

    }

    cpl = env->hflags & HF_CPL_MASK;

    if (!(env->cr[0] & CR0_PE_MASK) || cpl != 0) {

        raise_exception_err(EXCP0D_GPF, 0);

    }

    selector = (env->star >> 48) & 0xffff;

    if (env->hflags & HF_LMA_MASK) {

        if (dflag == 2) {

            cpu_x86_load_seg_cache(env, R_CS, (selector + 16) | 3,

                                   0, 0xffffffff,

                                   DESC_G_MASK | DESC_P_MASK |

                                   DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                                   DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK |

                                   DESC_L_MASK);

            env->eip = ECX;

        } else {

            cpu_x86_load_seg_cache(env, R_CS, selector | 3,

                                   0, 0xffffffff,

                                   DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                                   DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                                   DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK);

            env->eip = (uint32_t)ECX;

        }

        cpu_x86_load_seg_cache(env, R_SS, selector + 8,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_W_MASK | DESC_A_MASK);

        load_eflags((uint32_t)(env->regs[11]), TF_MASK | AC_MASK | ID_MASK |

                    IF_MASK | IOPL_MASK | VM_MASK | RF_MASK | NT_MASK);

        cpu_x86_set_cpl(env, 3);

    } else {

        cpu_x86_load_seg_cache(env, R_CS, selector | 3,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK);

        env->eip = (uint32_t)ECX;

        cpu_x86_load_seg_cache(env, R_SS, selector + 8,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK | (3 << DESC_DPL_SHIFT) |

                               DESC_W_MASK | DESC_A_MASK);

        env->eflags |= IF_MASK;

        cpu_x86_set_cpl(env, 3);

    }

#ifdef CONFIG_KQEMU

    if (kqemu_is_ok(env)) {

        if (env->hflags & HF_LMA_MASK)

            CC_OP = CC_OP_EFLAGS;

        env->exception_index = -1;

        cpu_loop_exit();

    }

#endif

}
