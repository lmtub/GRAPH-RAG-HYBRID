static int kvm_mce_in_exception(CPUState *env)

{

    struct kvm_msr_entry msr_mcg_status = {

        .index = MSR_MCG_STATUS,

    };

    int r;



    r = kvm_get_msr(env, &msr_mcg_status, 1);

    if (r == -1 || r == 0) {

        return -1;

    }

    return !!(msr_mcg_status.data & MCG_STATUS_MCIP);

}
