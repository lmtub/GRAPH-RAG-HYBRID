target_ulong helper_evpe(CPUMIPSState *env)

{

    CPUMIPSState *other_cpu = first_cpu;

    target_ulong prev = env->mvp->CP0_MVPControl;



    do {

        if (other_cpu != env

           /* If the VPE is WFI, don't disturb its sleep.  */

           && !mips_vpe_is_wfi(other_cpu)) {

            /* Enable the VPE.  */

            other_cpu->mvp->CP0_MVPControl |= (1 << CP0MVPCo_EVP);

            mips_vpe_wake(other_cpu); /* And wake it up.  */

        }

        other_cpu = other_cpu->next_cpu;

    } while (other_cpu);

    return prev;

}
