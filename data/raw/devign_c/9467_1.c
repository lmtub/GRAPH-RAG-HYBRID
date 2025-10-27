int do_balloon(Monitor *mon, const QDict *params,

	       MonitorCompletion cb, void *opaque)

{

    int ret;



    if (kvm_enabled() && !kvm_has_sync_mmu()) {

        qerror_report(QERR_KVM_MISSING_CAP, "synchronous MMU", "balloon");

        return -1;

    }



    ret = qemu_balloon(qdict_get_int(params, "value"), cb, opaque);

    if (ret == 0) {

        qerror_report(QERR_DEVICE_NOT_ACTIVE, "balloon");

        return -1;

    }



    cb(opaque, NULL);

    return 0;

}
