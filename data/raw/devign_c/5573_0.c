static void kvmppc_host_cpu_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    PowerPCCPUClass *pcc = POWERPC_CPU_CLASS(oc);

    uint32_t vmx = kvmppc_get_vmx();

    uint32_t dfp = kvmppc_get_dfp();

    uint32_t dcache_size = kvmppc_read_int_cpu_dt("d-cache-size");

    uint32_t icache_size = kvmppc_read_int_cpu_dt("i-cache-size");



    /* Now fix up the class with information we can query from the host */

    pcc->pvr = mfpvr();



    if (vmx != -1) {

        /* Only override when we know what the host supports */

        alter_insns(&pcc->insns_flags, PPC_ALTIVEC, vmx > 0);

        alter_insns(&pcc->insns_flags2, PPC2_VSX, vmx > 1);

    }

    if (dfp != -1) {

        /* Only override when we know what the host supports */

        alter_insns(&pcc->insns_flags2, PPC2_DFP, dfp);

    }



    if (dcache_size != -1) {

        pcc->l1_dcache_size = dcache_size;

    }



    if (icache_size != -1) {

        pcc->l1_icache_size = icache_size;

    }



    /* Reason: kvmppc_host_cpu_initfn() dies when !kvm_enabled() */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
