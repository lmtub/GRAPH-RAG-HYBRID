void vmexit(uint64_t exit_code, uint64_t exit_info_1)

{

    uint32_t int_ctl;



    if (loglevel & CPU_LOG_TB_IN_ASM)

        fprintf(logfile,"vmexit(%016" PRIx64 ", %016" PRIx64 ", %016" PRIx64 ", " TARGET_FMT_lx ")!\n",

                exit_code, exit_info_1,

                ldq_phys(env->vm_vmcb + offsetof(struct vmcb, control.exit_info_2)),

                EIP);



    if(env->hflags & HF_INHIBIT_IRQ_MASK) {

        stl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_state), SVM_INTERRUPT_SHADOW_MASK);

        env->hflags &= ~HF_INHIBIT_IRQ_MASK;

    } else {

        stl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_state), 0);

    }



    /* Save the VM state in the vmcb */

    SVM_SAVE_SEG(env->vm_vmcb, segs[R_ES], es);

    SVM_SAVE_SEG(env->vm_vmcb, segs[R_CS], cs);

    SVM_SAVE_SEG(env->vm_vmcb, segs[R_SS], ss);

    SVM_SAVE_SEG(env->vm_vmcb, segs[R_DS], ds);



    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.gdtr.base), env->gdt.base);

    stl_phys(env->vm_vmcb + offsetof(struct vmcb, save.gdtr.limit), env->gdt.limit);



    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.idtr.base), env->idt.base);

    stl_phys(env->vm_vmcb + offsetof(struct vmcb, save.idtr.limit), env->idt.limit);



    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.efer), env->efer);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr0), env->cr[0]);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr2), env->cr[2]);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr3), env->cr[3]);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr4), env->cr[4]);



    if ((int_ctl = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_ctl))) & V_INTR_MASKING_MASK) {

        int_ctl &= ~V_TPR_MASK;

        int_ctl |= env->cr[8] & V_TPR_MASK;

        stl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_ctl), int_ctl);

    }



    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rflags), compute_eflags());

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rip), env->eip);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rsp), ESP);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rax), EAX);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.dr7), env->dr[7]);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, save.dr6), env->dr[6]);

    stb_phys(env->vm_vmcb + offsetof(struct vmcb, save.cpl), env->hflags & HF_CPL_MASK);



    /* Reload the host state from vm_hsave */

    env->hflags &= ~HF_HIF_MASK;

    env->intercept = 0;

    env->intercept_exceptions = 0;

    env->interrupt_request &= ~CPU_INTERRUPT_VIRQ;



    env->gdt.base  = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.gdtr.base));

    env->gdt.limit = ldl_phys(env->vm_hsave + offsetof(struct vmcb, save.gdtr.limit));



    env->idt.base  = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.idtr.base));

    env->idt.limit = ldl_phys(env->vm_hsave + offsetof(struct vmcb, save.idtr.limit));



    cpu_x86_update_cr0(env, ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr0)) | CR0_PE_MASK);

    cpu_x86_update_cr4(env, ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr4)));

    cpu_x86_update_cr3(env, ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr3)));

    if (int_ctl & V_INTR_MASKING_MASK)

        env->cr[8] = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr8));

    /* we need to set the efer after the crs so the hidden flags get set properly */

#ifdef TARGET_X86_64

    env->efer  = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.efer));

    env->hflags &= ~HF_LMA_MASK;

    if (env->efer & MSR_EFER_LMA)

       env->hflags |= HF_LMA_MASK;

#endif



    env->eflags = 0;

    load_eflags(ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.rflags)),

                ~(CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C | DF_MASK));

    CC_OP = CC_OP_EFLAGS;



    SVM_LOAD_SEG(env->vm_hsave, ES, es);

    SVM_LOAD_SEG(env->vm_hsave, CS, cs);

    SVM_LOAD_SEG(env->vm_hsave, SS, ss);

    SVM_LOAD_SEG(env->vm_hsave, DS, ds);



    EIP = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.rip));

    ESP = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.rsp));

    EAX = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.rax));



    env->dr[6] = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.dr6));

    env->dr[7] = ldq_phys(env->vm_hsave + offsetof(struct vmcb, save.dr7));



    /* other setups */

    cpu_x86_set_cpl(env, 0);

    stl_phys(env->vm_vmcb + offsetof(struct vmcb, control.exit_code_hi), (uint32_t)(exit_code >> 32));

    stl_phys(env->vm_vmcb + offsetof(struct vmcb, control.exit_code), exit_code);

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, control.exit_info_1), exit_info_1);



    helper_clgi();

    /* FIXME: Resets the current ASID register to zero (host ASID). */



    /* Clears the V_IRQ and V_INTR_MASKING bits inside the processor. */



    /* Clears the TSC_OFFSET inside the processor. */



    /* If the host is in PAE mode, the processor reloads the host's PDPEs

       from the page table indicated the host's CR3. If the PDPEs contain

       illegal state, the processor causes a shutdown. */



    /* Forces CR0.PE = 1, RFLAGS.VM = 0. */

    env->cr[0] |= CR0_PE_MASK;

    env->eflags &= ~VM_MASK;



    /* Disables all breakpoints in the host DR7 register. */



    /* Checks the reloaded host state for consistency. */



    /* If the host's rIP reloaded by #VMEXIT is outside the limit of the

       host's code segment or non-canonical (in the case of long mode), a

       #GP fault is delivered inside the host.) */



    /* remove any pending exception */

    env->exception_index = -1;

    env->error_code = 0;

    env->old_exception = -1;



    regs_to_env();

    cpu_loop_exit();

}
