BalloonInfo *qmp_query_balloon(Error **errp)

{

    BalloonInfo *info;



    if (kvm_enabled() && !kvm_has_sync_mmu()) {

        error_set(errp, QERR_KVM_MISSING_CAP, "synchronous MMU", "balloon");

        return NULL;

    }



    info = g_malloc0(sizeof(*info));



    if (qemu_balloon_status(info) == 0) {

        error_set(errp, QERR_DEVICE_NOT_ACTIVE, "balloon");

        qapi_free_BalloonInfo(info);

        return NULL;

    }



    return info;

}
