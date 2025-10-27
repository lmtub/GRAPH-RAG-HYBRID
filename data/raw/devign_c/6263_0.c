void helper_vmrun(CPUX86State *env, int aflag, int next_eip_addend)

{

    CPUState *cs = CPU(x86_env_get_cpu(env));

    target_ulong addr;

    uint32_t event_inj;

    uint32_t int_ctl;



    cpu_svm_check_intercept_param(env, SVM_EXIT_VMRUN, 0);



    if (aflag == 2) {

        addr = env->regs[R_EAX];

    } else {

        addr = (uint32_t)env->regs[R_EAX];

    }



    qemu_log_mask(CPU_LOG_TB_IN_ASM, "vmrun! " TARGET_FMT_lx "\n", addr);



    env->vm_vmcb = addr;



    /* save the current CPU state in the hsave page */

    stq_phys(cs->as, env->vm_hsave + offsetof(struct vmcb, save.gdtr.base),

             env->gdt.base);

    stl_phys(cs->as, env->vm_hsave + offsetof(struct vmcb, save.gdtr.limit),

             env->gdt.limit);



    stq_phys(cs->as, env->vm_hsave + offsetof(struct vmcb, save.idtr.base),

             env->idt.base);

    stl_phys(cs->as, env->vm_hsave + offsetof(struct vmcb, save.idtr.limit),

             env->idt.limit);



    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.cr0), env->cr[0]);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.cr2), env->cr[2]);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.cr3), env->cr[3]);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.cr4), env->cr[4]);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.dr6), env->dr[6]);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.dr7), env->dr[7]);



    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.efer), env->efer);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.rflags),

             cpu_compute_eflags(env));



    svm_save_seg(env, env->vm_hsave + offsetof(struct vmcb, save.es),

                 &env->segs[R_ES]);

    svm_save_seg(env, env->vm_hsave + offsetof(struct vmcb, save.cs),

                 &env->segs[R_CS]);

    svm_save_seg(env, env->vm_hsave + offsetof(struct vmcb, save.ss),

                 &env->segs[R_SS]);

    svm_save_seg(env, env->vm_hsave + offsetof(struct vmcb, save.ds),

                 &env->segs[R_DS]);



    stq_phys(cs->as, env->vm_hsave + offsetof(struct vmcb, save.rip),

             env->eip + next_eip_addend);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.rsp), env->regs[R_ESP]);

    stq_phys(cs->as,

             env->vm_hsave + offsetof(struct vmcb, save.rax), env->regs[R_EAX]);



    /* load the interception bitmaps so we do not need to access the

       vmcb in svm mode */

    env->intercept = ldq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                                      control.intercept));

    env->intercept_cr_read = lduw_phys(cs->as, env->vm_vmcb +

                                       offsetof(struct vmcb,

                                                control.intercept_cr_read));

    env->intercept_cr_write = lduw_phys(cs->as, env->vm_vmcb +

                                        offsetof(struct vmcb,

                                                 control.intercept_cr_write));

    env->intercept_dr_read = lduw_phys(cs->as, env->vm_vmcb +

                                       offsetof(struct vmcb,

                                                control.intercept_dr_read));

    env->intercept_dr_write = lduw_phys(cs->as, env->vm_vmcb +

                                        offsetof(struct vmcb,

                                                 control.intercept_dr_write));

    env->intercept_exceptions = ldl_phys(cs->as, env->vm_vmcb +

                                         offsetof(struct vmcb,

                                                  control.intercept_exceptions

                                                  ));



    /* enable intercepts */

    env->hflags |= HF_SVMI_MASK;



    env->tsc_offset = ldq_phys(cs->as, env->vm_vmcb +

                               offsetof(struct vmcb, control.tsc_offset));



    env->gdt.base  = ldq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                                      save.gdtr.base));

    env->gdt.limit = ldl_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                                      save.gdtr.limit));



    env->idt.base  = ldq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                                      save.idtr.base));

    env->idt.limit = ldl_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                                      save.idtr.limit));



    /* clear exit_info_2 so we behave like the real hardware */

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, control.exit_info_2), 0);



    cpu_x86_update_cr0(env, ldq_phys(cs->as,

                                     env->vm_vmcb + offsetof(struct vmcb,

                                                             save.cr0)));

    cpu_x86_update_cr4(env, ldq_phys(cs->as,

                                     env->vm_vmcb + offsetof(struct vmcb,

                                                             save.cr4)));

    cpu_x86_update_cr3(env, ldq_phys(cs->as,

                                     env->vm_vmcb + offsetof(struct vmcb,

                                                             save.cr3)));

    env->cr[2] = ldq_phys(cs->as,

                          env->vm_vmcb + offsetof(struct vmcb, save.cr2));

    int_ctl = ldl_phys(cs->as,

                       env->vm_vmcb + offsetof(struct vmcb, control.int_ctl));

    env->hflags2 &= ~(HF2_HIF_MASK | HF2_VINTR_MASK);

    if (int_ctl & V_INTR_MASKING_MASK) {

        env->v_tpr = int_ctl & V_TPR_MASK;

        env->hflags2 |= HF2_VINTR_MASK;

        if (env->eflags & IF_MASK) {

            env->hflags2 |= HF2_HIF_MASK;

        }

    }



    cpu_load_efer(env,

                  ldq_phys(cs->as,

                           env->vm_vmcb + offsetof(struct vmcb, save.efer)));

    env->eflags = 0;

    cpu_load_eflags(env, ldq_phys(cs->as,

                                  env->vm_vmcb + offsetof(struct vmcb,

                                                          save.rflags)),

                    ~(CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C | DF_MASK));

    CC_OP = CC_OP_EFLAGS;



    svm_load_seg_cache(env, env->vm_vmcb + offsetof(struct vmcb, save.es),

                       R_ES);

    svm_load_seg_cache(env, env->vm_vmcb + offsetof(struct vmcb, save.cs),

                       R_CS);

    svm_load_seg_cache(env, env->vm_vmcb + offsetof(struct vmcb, save.ss),

                       R_SS);

    svm_load_seg_cache(env, env->vm_vmcb + offsetof(struct vmcb, save.ds),

                       R_DS);



    env->eip = ldq_phys(cs->as,

                        env->vm_vmcb + offsetof(struct vmcb, save.rip));



    env->regs[R_ESP] = ldq_phys(cs->as,

                                env->vm_vmcb + offsetof(struct vmcb, save.rsp));

    env->regs[R_EAX] = ldq_phys(cs->as,

                                env->vm_vmcb + offsetof(struct vmcb, save.rax));

    env->dr[7] = ldq_phys(cs->as,

                          env->vm_vmcb + offsetof(struct vmcb, save.dr7));

    env->dr[6] = ldq_phys(cs->as,

                          env->vm_vmcb + offsetof(struct vmcb, save.dr6));

    cpu_x86_set_cpl(env, ldub_phys(cs->as,

                                   env->vm_vmcb + offsetof(struct vmcb,

                                                           save.cpl)));



    /* FIXME: guest state consistency checks */



    switch (ldub_phys(cs->as,

                      env->vm_vmcb + offsetof(struct vmcb, control.tlb_ctl))) {

    case TLB_CONTROL_DO_NOTHING:

        break;

    case TLB_CONTROL_FLUSH_ALL_ASID:

        /* FIXME: this is not 100% correct but should work for now */

        tlb_flush(cs, 1);

        break;

    }



    env->hflags2 |= HF2_GIF_MASK;



    if (int_ctl & V_IRQ_MASK) {

        CPUState *cs = CPU(x86_env_get_cpu(env));



        cs->interrupt_request |= CPU_INTERRUPT_VIRQ;

    }



    /* maybe we need to inject an event */

    event_inj = ldl_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                                 control.event_inj));

    if (event_inj & SVM_EVTINJ_VALID) {

        uint8_t vector = event_inj & SVM_EVTINJ_VEC_MASK;

        uint16_t valid_err = event_inj & SVM_EVTINJ_VALID_ERR;

        uint32_t event_inj_err = ldl_phys(cs->as, env->vm_vmcb +

                                          offsetof(struct vmcb,

                                                   control.event_inj_err));



        qemu_log_mask(CPU_LOG_TB_IN_ASM, "Injecting(%#hx): ", valid_err);

        /* FIXME: need to implement valid_err */

        switch (event_inj & SVM_EVTINJ_TYPE_MASK) {

        case SVM_EVTINJ_TYPE_INTR:

            cs->exception_index = vector;

            env->error_code = event_inj_err;

            env->exception_is_int = 0;

            env->exception_next_eip = -1;

            qemu_log_mask(CPU_LOG_TB_IN_ASM, "INTR");

            /* XXX: is it always correct? */

            do_interrupt_x86_hardirq(env, vector, 1);

            break;

        case SVM_EVTINJ_TYPE_NMI:

            cs->exception_index = EXCP02_NMI;

            env->error_code = event_inj_err;

            env->exception_is_int = 0;

            env->exception_next_eip = env->eip;

            qemu_log_mask(CPU_LOG_TB_IN_ASM, "NMI");

            cpu_loop_exit(cs);

            break;

        case SVM_EVTINJ_TYPE_EXEPT:

            cs->exception_index = vector;

            env->error_code = event_inj_err;

            env->exception_is_int = 0;

            env->exception_next_eip = -1;

            qemu_log_mask(CPU_LOG_TB_IN_ASM, "EXEPT");

            cpu_loop_exit(cs);

            break;

        case SVM_EVTINJ_TYPE_SOFT:

            cs->exception_index = vector;

            env->error_code = event_inj_err;

            env->exception_is_int = 1;

            env->exception_next_eip = env->eip;

            qemu_log_mask(CPU_LOG_TB_IN_ASM, "SOFT");

            cpu_loop_exit(cs);

            break;

        }

        qemu_log_mask(CPU_LOG_TB_IN_ASM, " %#x %#x\n", cs->exception_index,

                      env->error_code);

    }

}
