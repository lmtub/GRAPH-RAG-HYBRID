void do_store_msr (CPUPPCState *env, target_ulong value)

{

    int enter_pm;



    value &= env->msr_mask;

    if (((value >> MSR_IR) & 1) != msr_ir ||

        ((value >> MSR_DR) & 1) != msr_dr) {

        /* Flush all tlb when changing translation mode */

        tlb_flush(env, 1);

        env->interrupt_request |= CPU_INTERRUPT_EXITTB;

    }

#if 0

    if (loglevel != 0) {

        fprintf(logfile, "%s: T0 %08lx\n", __func__, value);

    }

#endif

    switch (PPC_EXCP(env)) {

    case PPC_FLAGS_EXCP_602:

    case PPC_FLAGS_EXCP_603:

        if (((value >> MSR_TGPR) & 1) != msr_tgpr) {

            /* Swap temporary saved registers with GPRs */

            swap_gpr_tgpr(env);

        }

        break;

    default:

        break;

    }

#if defined (TARGET_PPC64)

    msr_sf   = (value >> MSR_SF)   & 1;

    msr_isf  = (value >> MSR_ISF)  & 1;

    msr_hv   = (value >> MSR_HV)   & 1;

#endif

    msr_ucle = (value >> MSR_UCLE) & 1;

    msr_vr   = (value >> MSR_VR)   & 1; /* VR / SPE */

    msr_ap   = (value >> MSR_AP)   & 1;

    msr_sa   = (value >> MSR_SA)   & 1;

    msr_key  = (value >> MSR_KEY)  & 1;

    msr_pow  = (value >> MSR_POW)  & 1; /* POW / WE */

    msr_tlb  = (value >> MSR_TLB)  & 1; /* TLB / TGPR / CE */

    msr_ile  = (value >> MSR_ILE)  & 1;

    msr_ee   = (value >> MSR_EE)   & 1;

    msr_pr   = (value >> MSR_PR)   & 1;

    msr_fp   = (value >> MSR_FP)   & 1;

    msr_me   = (value >> MSR_ME)   & 1;

    msr_fe0  = (value >> MSR_FE0)  & 1;

    msr_se   = (value >> MSR_SE)   & 1; /* SE / DWE / UBLE */

    msr_be   = (value >> MSR_BE)   & 1; /* BE / DE */

    msr_fe1  = (value >> MSR_FE1)  & 1;

    msr_al   = (value >> MSR_AL)   & 1;

    msr_ip   = (value >> MSR_IP)   & 1;

    msr_ir   = (value >> MSR_IR)   & 1; /* IR / IS */

    msr_dr   = (value >> MSR_DR)   & 1; /* DR / DS */

    msr_pe   = (value >> MSR_PE)   & 1; /* PE / EP */

    msr_px   = (value >> MSR_PX)   & 1; /* PX / PMM */

    msr_ri   = (value >> MSR_RI)   & 1;

    msr_le   = (value >> MSR_LE)   & 1;

    do_compute_hflags(env);



    enter_pm = 0;

    switch (PPC_EXCP(env)) {

    case PPC_FLAGS_EXCP_603:

        /* Don't handle SLEEP mode: we should disable all clocks...

         * No dynamic power-management.

         */

        if (msr_pow == 1 && (env->spr[SPR_HID0] & 0x00C00000) != 0)

            enter_pm = 1;

        break;

    case PPC_FLAGS_EXCP_604:

        if (msr_pow == 1)

            enter_pm = 1;

        break;

    case PPC_FLAGS_EXCP_7x0:

        if (msr_pow == 1 && (env->spr[SPR_HID0] & 0x00E00000) != 0)

            enter_pm = 1;

        break;

    default:

        break;

    }

    if (enter_pm) {

        /* power save: exit cpu loop */

        env->halted = 1;

        env->exception_index = EXCP_HLT;

        cpu_loop_exit();

    }

}
