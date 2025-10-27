void colo_do_failover(MigrationState *s)

{

    /* Make sure VM stopped while failover happened. */

    if (!colo_runstate_is_stopped()) {

        vm_stop_force_state(RUN_STATE_COLO);

    }



    if (get_colo_mode() == COLO_MODE_PRIMARY) {

        primary_vm_do_failover();



    }

}