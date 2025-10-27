void do_smm_enter(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    CPUState *cs = CPU(cpu);

    target_ulong sm_state;

    SegmentCache *dt;

    int i, offset;



    qemu_log_mask(CPU_LOG_INT, "SMM: enter\n");

    log_cpu_state_mask(CPU_LOG_INT, CPU(cpu), CPU_DUMP_CCOP);



    env->hflags |= HF_SMM_MASK;

    if (env->hflags2 & HF2_NMI_MASK) {

        env->hflags2 |= HF2_SMM_INSIDE_NMI_MASK;

    } else {

        env->hflags2 |= HF2_NMI_MASK;

    }

    cpu_smm_update(env);



    sm_state = env->smbase + 0x8000;



#ifdef TARGET_X86_64

    for (i = 0; i < 6; i++) {

        dt = &env->segs[i];

        offset = 0x7e00 + i * 16;

        x86_stw_phys(cs, sm_state + offset, dt->selector);

        x86_stw_phys(cs, sm_state + offset + 2, (dt->flags >> 8) & 0xf0ff);

        x86_stl_phys(cs, sm_state + offset + 4, dt->limit);

        x86_stq_phys(cs, sm_state + offset + 8, dt->base);

    }



    x86_stq_phys(cs, sm_state + 0x7e68, env->gdt.base);

    x86_stl_phys(cs, sm_state + 0x7e64, env->gdt.limit);



    x86_stw_phys(cs, sm_state + 0x7e70, env->ldt.selector);

    x86_stq_phys(cs, sm_state + 0x7e78, env->ldt.base);

    x86_stl_phys(cs, sm_state + 0x7e74, env->ldt.limit);

    x86_stw_phys(cs, sm_state + 0x7e72, (env->ldt.flags >> 8) & 0xf0ff);



    x86_stq_phys(cs, sm_state + 0x7e88, env->idt.base);

    x86_stl_phys(cs, sm_state + 0x7e84, env->idt.limit);



    x86_stw_phys(cs, sm_state + 0x7e90, env->tr.selector);

    x86_stq_phys(cs, sm_state + 0x7e98, env->tr.base);

    x86_stl_phys(cs, sm_state + 0x7e94, env->tr.limit);

    x86_stw_phys(cs, sm_state + 0x7e92, (env->tr.flags >> 8) & 0xf0ff);



    x86_stq_phys(cs, sm_state + 0x7ed0, env->efer);



    x86_stq_phys(cs, sm_state + 0x7ff8, env->regs[R_EAX]);

    x86_stq_phys(cs, sm_state + 0x7ff0, env->regs[R_ECX]);

    x86_stq_phys(cs, sm_state + 0x7fe8, env->regs[R_EDX]);

    x86_stq_phys(cs, sm_state + 0x7fe0, env->regs[R_EBX]);

    x86_stq_phys(cs, sm_state + 0x7fd8, env->regs[R_ESP]);

    x86_stq_phys(cs, sm_state + 0x7fd0, env->regs[R_EBP]);

    x86_stq_phys(cs, sm_state + 0x7fc8, env->regs[R_ESI]);

    x86_stq_phys(cs, sm_state + 0x7fc0, env->regs[R_EDI]);

    for (i = 8; i < 16; i++) {

        x86_stq_phys(cs, sm_state + 0x7ff8 - i * 8, env->regs[i]);

    }

    x86_stq_phys(cs, sm_state + 0x7f78, env->eip);

    x86_stl_phys(cs, sm_state + 0x7f70, cpu_compute_eflags(env));

    x86_stl_phys(cs, sm_state + 0x7f68, env->dr[6]);

    x86_stl_phys(cs, sm_state + 0x7f60, env->dr[7]);



    x86_stl_phys(cs, sm_state + 0x7f48, env->cr[4]);

    x86_stq_phys(cs, sm_state + 0x7f50, env->cr[3]);

    x86_stl_phys(cs, sm_state + 0x7f58, env->cr[0]);



    x86_stl_phys(cs, sm_state + 0x7efc, SMM_REVISION_ID);

    x86_stl_phys(cs, sm_state + 0x7f00, env->smbase);

#else

    x86_stl_phys(cs, sm_state + 0x7ffc, env->cr[0]);

    x86_stl_phys(cs, sm_state + 0x7ff8, env->cr[3]);

    x86_stl_phys(cs, sm_state + 0x7ff4, cpu_compute_eflags(env));

    x86_stl_phys(cs, sm_state + 0x7ff0, env->eip);

    x86_stl_phys(cs, sm_state + 0x7fec, env->regs[R_EDI]);

    x86_stl_phys(cs, sm_state + 0x7fe8, env->regs[R_ESI]);

    x86_stl_phys(cs, sm_state + 0x7fe4, env->regs[R_EBP]);

    x86_stl_phys(cs, sm_state + 0x7fe0, env->regs[R_ESP]);

    x86_stl_phys(cs, sm_state + 0x7fdc, env->regs[R_EBX]);

    x86_stl_phys(cs, sm_state + 0x7fd8, env->regs[R_EDX]);

    x86_stl_phys(cs, sm_state + 0x7fd4, env->regs[R_ECX]);

    x86_stl_phys(cs, sm_state + 0x7fd0, env->regs[R_EAX]);

    x86_stl_phys(cs, sm_state + 0x7fcc, env->dr[6]);

    x86_stl_phys(cs, sm_state + 0x7fc8, env->dr[7]);



    x86_stl_phys(cs, sm_state + 0x7fc4, env->tr.selector);

    x86_stl_phys(cs, sm_state + 0x7f64, env->tr.base);

    x86_stl_phys(cs, sm_state + 0x7f60, env->tr.limit);

    x86_stl_phys(cs, sm_state + 0x7f5c, (env->tr.flags >> 8) & 0xf0ff);



    x86_stl_phys(cs, sm_state + 0x7fc0, env->ldt.selector);

    x86_stl_phys(cs, sm_state + 0x7f80, env->ldt.base);

    x86_stl_phys(cs, sm_state + 0x7f7c, env->ldt.limit);

    x86_stl_phys(cs, sm_state + 0x7f78, (env->ldt.flags >> 8) & 0xf0ff);



    x86_stl_phys(cs, sm_state + 0x7f74, env->gdt.base);

    x86_stl_phys(cs, sm_state + 0x7f70, env->gdt.limit);



    x86_stl_phys(cs, sm_state + 0x7f58, env->idt.base);

    x86_stl_phys(cs, sm_state + 0x7f54, env->idt.limit);



    for (i = 0; i < 6; i++) {

        dt = &env->segs[i];

        if (i < 3) {

            offset = 0x7f84 + i * 12;

        } else {

            offset = 0x7f2c + (i - 3) * 12;

        }

        x86_stl_phys(cs, sm_state + 0x7fa8 + i * 4, dt->selector);

        x86_stl_phys(cs, sm_state + offset + 8, dt->base);

        x86_stl_phys(cs, sm_state + offset + 4, dt->limit);

        x86_stl_phys(cs, sm_state + offset, (dt->flags >> 8) & 0xf0ff);

    }

    x86_stl_phys(cs, sm_state + 0x7f14, env->cr[4]);



    x86_stl_phys(cs, sm_state + 0x7efc, SMM_REVISION_ID);

    x86_stl_phys(cs, sm_state + 0x7ef8, env->smbase);

#endif

    /* init SMM cpu state */



#ifdef TARGET_X86_64

    cpu_load_efer(env, 0);

#endif

    cpu_load_eflags(env, 0, ~(CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C |

                              DF_MASK));

    env->eip = 0x00008000;

    cpu_x86_update_cr0(env,

                       env->cr[0] & ~(CR0_PE_MASK | CR0_EM_MASK | CR0_TS_MASK |

                                      CR0_PG_MASK));

    cpu_x86_update_cr4(env, 0);

    env->dr[7] = 0x00000400;



    cpu_x86_load_seg_cache(env, R_CS, (env->smbase >> 4) & 0xffff, env->smbase,

                           0xffffffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_DS, 0, 0, 0xffffffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_ES, 0, 0, 0xffffffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_SS, 0, 0, 0xffffffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_FS, 0, 0, 0xffffffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_GS, 0, 0, 0xffffffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

}
