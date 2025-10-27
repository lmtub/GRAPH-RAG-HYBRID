void qmp_balloon(int64_t value, Error **errp)

{

    if (kvm_enabled() && !kvm_has_sync_mmu()) {

        error_set(errp, QERR_KVM_MISSING_CAP, "synchronous MMU", "balloon");

        return;

    }



    if (value <= 0) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "target", "a size");

        return;

    }

    

    if (qemu_balloon(value) == 0) {

        error_set(errp, QERR_DEVICE_NOT_ACTIVE, "balloon");

    }

}
