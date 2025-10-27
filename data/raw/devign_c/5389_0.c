static void virtio_scsi_migration_state_changed(Notifier *notifier, void *data)

{

    VirtIOSCSI *s = container_of(notifier, VirtIOSCSI,

                                 migration_state_notifier);

    MigrationState *mig = data;



    if (migration_in_setup(mig)) {

        if (!s->dataplane_started) {

            return;

        }

        virtio_scsi_dataplane_stop(s);

        s->dataplane_disabled = true;

    } else if (migration_has_finished(mig) ||

               migration_has_failed(mig)) {

        if (s->dataplane_started) {

            return;

        }

        bdrv_drain_all(); /* complete in-flight non-dataplane requests */

        s->dataplane_disabled = false;

    }

}
