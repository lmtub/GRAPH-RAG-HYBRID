void helper_rdmsr(void)

{

    uint64_t val;



    helper_svm_check_intercept_param(SVM_EXIT_MSR, 0);



    switch((uint32_t)ECX) {

    case MSR_IA32_SYSENTER_CS:

        val = env->sysenter_cs;


    case MSR_IA32_SYSENTER_ESP:

        val = env->sysenter_esp;


    case MSR_IA32_SYSENTER_EIP:

        val = env->sysenter_eip;


    case MSR_IA32_APICBASE:

        val = cpu_get_apic_base(env);


    case MSR_EFER:

        val = env->efer;


    case MSR_STAR:

        val = env->star;


    case MSR_PAT:

        val = env->pat;


    case MSR_VM_HSAVE_PA:

        val = env->vm_hsave;


    case MSR_IA32_PERF_STATUS:

        /* tsc_increment_by_tick */

        val = 1000ULL;

        /* CPU multiplier */

        val |= (((uint64_t)4ULL) << 40);


#ifdef TARGET_X86_64

    case MSR_LSTAR:

        val = env->lstar;


    case MSR_CSTAR:

        val = env->cstar;


    case MSR_FMASK:

        val = env->fmask;


    case MSR_FSBASE:

        val = env->segs[R_FS].base;


    case MSR_GSBASE:

        val = env->segs[R_GS].base;


    case MSR_KERNELGSBASE:

        val = env->kernelgsbase;


#endif

#ifdef USE_KQEMU

    case MSR_QPI_COMMBASE:

        if (env->kqemu_enabled) {

            val = kqemu_comm_base;

        } else {

            val = 0;

        }


#endif









































    default:

        /* XXX: exception ? */

        val = 0;


    }

    EAX = (uint32_t)(val);

    EDX = (uint32_t)(val >> 32);

}