void ppc_set_compat(PowerPCCPU *cpu, uint32_t compat_pvr, Error **errp)

{

    int ret = 0;

    CPUPPCState *env = &cpu->env;

    PowerPCCPUClass *host_pcc;



    cpu->compat_pvr = compat_pvr;



    switch (compat_pvr) {

    case CPU_POWERPC_LOGICAL_2_05:

        env->spr[SPR_PCR] = PCR_TM_DIS | PCR_VSX_DIS | PCR_COMPAT_2_07 |

                            PCR_COMPAT_2_06 | PCR_COMPAT_2_05;

        break;

    case CPU_POWERPC_LOGICAL_2_06:

    case CPU_POWERPC_LOGICAL_2_06_PLUS:

        env->spr[SPR_PCR] = PCR_TM_DIS | PCR_COMPAT_2_07 | PCR_COMPAT_2_06;

        break;

    case CPU_POWERPC_LOGICAL_2_07:

        env->spr[SPR_PCR] = PCR_COMPAT_2_07;

        break;

    default:

        env->spr[SPR_PCR] = 0;

        break;

    }



    host_pcc = kvm_ppc_get_host_cpu_class();

    if (host_pcc) {

        env->spr[SPR_PCR] &= host_pcc->pcr_mask;

    }



    if (kvm_enabled()) {

        ret = kvmppc_set_compat(cpu, cpu->compat_pvr);

        if (ret < 0) {

            error_setg_errno(errp, -ret,

                             "Unable to set CPU compatibility mode in KVM");

        }

    }

}
