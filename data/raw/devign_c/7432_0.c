void kvm_s390_apply_cpu_model(const S390CPUModel *model, Error **errp)

{

    struct kvm_s390_vm_cpu_processor prop  = {

        .fac_list = { 0 },

    };

    struct kvm_device_attr attr = {

        .group = KVM_S390_VM_CPU_MODEL,

        .attr = KVM_S390_VM_CPU_PROCESSOR,

        .addr = (uint64_t) &prop,

    };

    int rc;



    if (!model) {

        /* compatibility handling if cpu models are disabled */

        if (kvm_s390_cmma_available() && !mem_path) {

            kvm_s390_enable_cmma();

        }

        return;

    }

    if (!kvm_s390_cpu_models_supported()) {

        error_setg(errp, "KVM doesn't support CPU models");

        return;

    }

    prop.cpuid = s390_cpuid_from_cpu_model(model);

    prop.ibc = s390_ibc_from_cpu_model(model);

    /* configure cpu features indicated via STFL(e) */

    s390_fill_feat_block(model->features, S390_FEAT_TYPE_STFL,

                         (uint8_t *) prop.fac_list);

    rc = kvm_vm_ioctl(kvm_state, KVM_SET_DEVICE_ATTR, &attr);

    if (rc) {

        error_setg(errp, "KVM: Error configuring the CPU model: %d", rc);

        return;

    }

    /* configure cpu features indicated e.g. via SCLP */

    rc = configure_cpu_feat(model->features);

    if (rc) {

        error_setg(errp, "KVM: Error configuring CPU features: %d", rc);

        return;

    }

    /* configure cpu subfunctions indicated via query / test bit */

    rc = configure_cpu_subfunc(model->features);

    if (rc) {

        error_setg(errp, "KVM: Error configuring CPU subfunctions: %d", rc);

        return;

    }

    /* enable CMM via CMMA - disable on hugetlbfs */

    if (test_bit(S390_FEAT_CMM, model->features)) {

        if (mem_path) {

            error_report("Warning: CMM will not be enabled because it is not "

                         "compatible to hugetlbfs.");

        } else {

            kvm_s390_enable_cmma();

        }

    }

}
