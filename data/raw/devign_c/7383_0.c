void ppc_hash64_set_external_hpt(PowerPCCPU *cpu, void *hpt, int shift,

                                 Error **errp)

{

    CPUPPCState *env = &cpu->env;

    Error *local_err = NULL;



    cpu_synchronize_state(CPU(cpu));



    env->external_htab = hpt;

    ppc_hash64_set_sdr1(cpu, (target_ulong)(uintptr_t)hpt | (shift - 18),

                        &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    /* Not strictly necessary, but makes it clearer that an external

     * htab is in use when debugging */

    env->htab_base = -1;



    if (kvm_enabled()) {

        if (kvmppc_put_books_sregs(cpu) < 0) {

            error_setg(errp, "Unable to update SDR1 in KVM");

        }

    }

}
