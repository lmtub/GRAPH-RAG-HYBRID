static void handle_event(int event)

{

    static bool logged;



    if (event & ~PVPANIC_PANICKED && !logged) {

        qemu_log_mask(LOG_GUEST_ERROR, "pvpanic: unknown event %#x.\n", event);

        logged = true;

    }



    if (event & PVPANIC_PANICKED) {

        panicked_mon_event("pause");

        vm_stop(RUN_STATE_GUEST_PANICKED);

        return;

    }

}
