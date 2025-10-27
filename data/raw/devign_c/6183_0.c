int hvf_vcpu_exec(CPUState *cpu)

{

    X86CPU *x86_cpu = X86_CPU(cpu);

    CPUX86State *env = &x86_cpu->env;

    int ret = 0;

    uint64_t rip = 0;



    cpu->halted = 0;



    if (hvf_process_events(cpu)) {

        return EXCP_HLT;

    }



    do {

        if (cpu->vcpu_dirty) {

            hvf_put_registers(cpu);

            cpu->vcpu_dirty = false;

        }



        env->hvf_emul->interruptable =

            !(rvmcs(cpu->hvf_fd, VMCS_GUEST_INTERRUPTIBILITY) &

             (VMCS_INTERRUPTIBILITY_STI_BLOCKING |

             VMCS_INTERRUPTIBILITY_MOVSS_BLOCKING));



        hvf_inject_interrupts(cpu);

        vmx_update_tpr(cpu);



        qemu_mutex_unlock_iothread();

        if (!cpu_is_bsp(X86_CPU(cpu)) && cpu->halted) {

            qemu_mutex_lock_iothread();

            return EXCP_HLT;

        }



        hv_return_t r  = hv_vcpu_run(cpu->hvf_fd);

        assert_hvf_ok(r);



        /* handle VMEXIT */

        uint64_t exit_reason = rvmcs(cpu->hvf_fd, VMCS_EXIT_REASON);

        uint64_t exit_qual = rvmcs(cpu->hvf_fd, VMCS_EXIT_QUALIFICATION);

        uint32_t ins_len = (uint32_t)rvmcs(cpu->hvf_fd,

                                           VMCS_EXIT_INSTRUCTION_LENGTH);

        uint64_t idtvec_info = rvmcs(cpu->hvf_fd, VMCS_IDT_VECTORING_INFO);

        rip = rreg(cpu->hvf_fd, HV_X86_RIP);

        RFLAGS(env) = rreg(cpu->hvf_fd, HV_X86_RFLAGS);

        env->eflags = RFLAGS(env);



        qemu_mutex_lock_iothread();



        update_apic_tpr(cpu);

        current_cpu = cpu;



        ret = 0;

        switch (exit_reason) {

        case EXIT_REASON_HLT: {

            macvm_set_rip(cpu, rip + ins_len);

            if (!((cpu->interrupt_request & CPU_INTERRUPT_HARD) &&

                (EFLAGS(env) & IF_MASK))

                && !(cpu->interrupt_request & CPU_INTERRUPT_NMI) &&

                !(idtvec_info & VMCS_IDT_VEC_VALID)) {

                cpu->halted = 1;

                ret = EXCP_HLT;

            }

            ret = EXCP_INTERRUPT;

            break;

        }

        case EXIT_REASON_MWAIT: {

            ret = EXCP_INTERRUPT;

            break;

        }

            /* Need to check if MMIO or unmmaped fault */

        case EXIT_REASON_EPT_FAULT:

        {

            hvf_slot *slot;

            addr_t gpa = rvmcs(cpu->hvf_fd, VMCS_GUEST_PHYSICAL_ADDRESS);



            if (((idtvec_info & VMCS_IDT_VEC_VALID) == 0) &&

                ((exit_qual & EXIT_QUAL_NMIUDTI) != 0)) {

                vmx_set_nmi_blocking(cpu);

            }



            slot = hvf_find_overlap_slot(gpa, gpa);

            /* mmio */

            if (ept_emulation_fault(slot, gpa, exit_qual)) {

                struct x86_decode decode;



                load_regs(cpu);

                env->hvf_emul->fetch_rip = rip;



                decode_instruction(env, &decode);

                exec_instruction(env, &decode);

                store_regs(cpu);

                break;

            }

            break;

        }

        case EXIT_REASON_INOUT:

        {

            uint32_t in = (exit_qual & 8) != 0;

            uint32_t size =  (exit_qual & 7) + 1;

            uint32_t string =  (exit_qual & 16) != 0;

            uint32_t port =  exit_qual >> 16;

            /*uint32_t rep = (exit_qual & 0x20) != 0;*/



#if 1

            if (!string && in) {

                uint64_t val = 0;

                load_regs(cpu);

                hvf_handle_io(env, port, &val, 0, size, 1);

                if (size == 1) {

                    AL(env) = val;

                } else if (size == 2) {

                    AX(env) = val;

                } else if (size == 4) {

                    RAX(env) = (uint32_t)val;

                } else {

                    VM_PANIC("size");

                }

                RIP(env) += ins_len;

                store_regs(cpu);

                break;

            } else if (!string && !in) {

                RAX(env) = rreg(cpu->hvf_fd, HV_X86_RAX);

                hvf_handle_io(env, port, &RAX(env), 1, size, 1);

                macvm_set_rip(cpu, rip + ins_len);

                break;

            }

#endif

            struct x86_decode decode;



            load_regs(cpu);

            env->hvf_emul->fetch_rip = rip;



            decode_instruction(env, &decode);

            VM_PANIC_ON(ins_len != decode.len);

            exec_instruction(env, &decode);

            store_regs(cpu);



            break;

        }

        case EXIT_REASON_CPUID: {

            uint32_t rax = (uint32_t)rreg(cpu->hvf_fd, HV_X86_RAX);

            uint32_t rbx = (uint32_t)rreg(cpu->hvf_fd, HV_X86_RBX);

            uint32_t rcx = (uint32_t)rreg(cpu->hvf_fd, HV_X86_RCX);

            uint32_t rdx = (uint32_t)rreg(cpu->hvf_fd, HV_X86_RDX);



            cpu_x86_cpuid(env, rax, rcx, &rax, &rbx, &rcx, &rdx);



            wreg(cpu->hvf_fd, HV_X86_RAX, rax);

            wreg(cpu->hvf_fd, HV_X86_RBX, rbx);

            wreg(cpu->hvf_fd, HV_X86_RCX, rcx);

            wreg(cpu->hvf_fd, HV_X86_RDX, rdx);



            macvm_set_rip(cpu, rip + ins_len);

            break;

        }

        case EXIT_REASON_XSETBV: {

            X86CPU *x86_cpu = X86_CPU(cpu);

            CPUX86State *env = &x86_cpu->env;

            uint32_t eax = (uint32_t)rreg(cpu->hvf_fd, HV_X86_RAX);

            uint32_t ecx = (uint32_t)rreg(cpu->hvf_fd, HV_X86_RCX);

            uint32_t edx = (uint32_t)rreg(cpu->hvf_fd, HV_X86_RDX);



            if (ecx) {

                macvm_set_rip(cpu, rip + ins_len);

                break;

            }

            env->xcr0 = ((uint64_t)edx << 32) | eax;

            wreg(cpu->hvf_fd, HV_X86_XCR0, env->xcr0 | 1);

            macvm_set_rip(cpu, rip + ins_len);

            break;

        }

        case EXIT_REASON_INTR_WINDOW:

            vmx_clear_int_window_exiting(cpu);

            ret = EXCP_INTERRUPT;

            break;

        case EXIT_REASON_NMI_WINDOW:

            vmx_clear_nmi_window_exiting(cpu);

            ret = EXCP_INTERRUPT;

            break;

        case EXIT_REASON_EXT_INTR:

            /* force exit and allow io handling */

            ret = EXCP_INTERRUPT;

            break;

        case EXIT_REASON_RDMSR:

        case EXIT_REASON_WRMSR:

        {

            load_regs(cpu);

            if (exit_reason == EXIT_REASON_RDMSR) {

                simulate_rdmsr(cpu);

            } else {

                simulate_wrmsr(cpu);

            }

            RIP(env) += rvmcs(cpu->hvf_fd, VMCS_EXIT_INSTRUCTION_LENGTH);

            store_regs(cpu);

            break;

        }

        case EXIT_REASON_CR_ACCESS: {

            int cr;

            int reg;



            load_regs(cpu);

            cr = exit_qual & 15;

            reg = (exit_qual >> 8) & 15;



            switch (cr) {

            case 0x0: {

                macvm_set_cr0(cpu->hvf_fd, RRX(env, reg));

                break;

            }

            case 4: {

                macvm_set_cr4(cpu->hvf_fd, RRX(env, reg));

                break;

            }

            case 8: {

                X86CPU *x86_cpu = X86_CPU(cpu);

                if (exit_qual & 0x10) {

                    RRX(env, reg) = cpu_get_apic_tpr(x86_cpu->apic_state);

                } else {

                    int tpr = RRX(env, reg);

                    cpu_set_apic_tpr(x86_cpu->apic_state, tpr);

                    ret = EXCP_INTERRUPT;

                }

                break;

            }

            default:

                error_report("Unrecognized CR %d\n", cr);

                abort();

            }

            RIP(env) += ins_len;

            store_regs(cpu);

            break;

        }

        case EXIT_REASON_APIC_ACCESS: { /* TODO */

            struct x86_decode decode;



            load_regs(cpu);

            env->hvf_emul->fetch_rip = rip;



            decode_instruction(env, &decode);

            exec_instruction(env, &decode);

            store_regs(cpu);

            break;

        }

        case EXIT_REASON_TPR: {

            ret = 1;

            break;

        }

        case EXIT_REASON_TASK_SWITCH: {

            uint64_t vinfo = rvmcs(cpu->hvf_fd, VMCS_IDT_VECTORING_INFO);

            x68_segment_selector sel = {.sel = exit_qual & 0xffff};

            vmx_handle_task_switch(cpu, sel, (exit_qual >> 30) & 0x3,

             vinfo & VMCS_INTR_VALID, vinfo & VECTORING_INFO_VECTOR_MASK, vinfo

             & VMCS_INTR_T_MASK);

            break;

        }

        case EXIT_REASON_TRIPLE_FAULT: {

            qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET);

            ret = EXCP_INTERRUPT;

            break;

        }

        case EXIT_REASON_RDPMC:

            wreg(cpu->hvf_fd, HV_X86_RAX, 0);

            wreg(cpu->hvf_fd, HV_X86_RDX, 0);

            macvm_set_rip(cpu, rip + ins_len);

            break;

        case VMX_REASON_VMCALL:

            /* TODO: inject #GP fault */

            break;

        default:

            error_report("%llx: unhandled exit %llx\n", rip, exit_reason);

        }

    } while (ret == 0);



    return ret;

}
