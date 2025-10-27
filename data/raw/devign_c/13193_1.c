void helper_wrmsr(void)

{

    uint64_t val;



    helper_svm_check_intercept_param(SVM_EXIT_MSR, 1);



    val = ((uint32_t)EAX) | ((uint64_t)((uint32_t)EDX) << 32);



    switch((uint32_t)ECX) {

    case MSR_IA32_SYSENTER_CS:

        env->sysenter_cs = val & 0xffff;


    case MSR_IA32_SYSENTER_ESP:

        env->sysenter_esp = val;


    case MSR_IA32_SYSENTER_EIP:

        env->sysenter_eip = val;


    case MSR_IA32_APICBASE:

        cpu_set_apic_base(env, val);


    case MSR_EFER:

        {

            uint64_t update_mask;

            update_mask = 0;

            if (env->cpuid_ext2_features & CPUID_EXT2_SYSCALL)

                update_mask |= MSR_EFER_SCE;

            if (env->cpuid_ext2_features & CPUID_EXT2_LM)

                update_mask |= MSR_EFER_LME;

            if (env->cpuid_ext2_features & CPUID_EXT2_FFXSR)

                update_mask |= MSR_EFER_FFXSR;

            if (env->cpuid_ext2_features & CPUID_EXT2_NX)

                update_mask |= MSR_EFER_NXE;

            if (env->cpuid_ext3_features & CPUID_EXT3_SVM)

                update_mask |= MSR_EFER_SVME;

            cpu_load_efer(env, (env->efer & ~update_mask) |

                          (val & update_mask));

        }


    case MSR_STAR:

        env->star = val;


    case MSR_PAT:

        env->pat = val;


    case MSR_VM_HSAVE_PA:

        env->vm_hsave = val;


#ifdef TARGET_X86_64

    case MSR_LSTAR:

        env->lstar = val;


    case MSR_CSTAR:

        env->cstar = val;


    case MSR_FMASK:

        env->fmask = val;


    case MSR_FSBASE:

        env->segs[R_FS].base = val;


    case MSR_GSBASE:

        env->segs[R_GS].base = val;


    case MSR_KERNELGSBASE:

        env->kernelgsbase = val;


#endif









































    default:

        /* XXX: exception ? */


    }

}