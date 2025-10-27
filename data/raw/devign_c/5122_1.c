static void guest_fsfreeze_cleanup(void)

{

    int64_t ret;

    Error *err = NULL;



    if (ga_is_frozen(ga_state) == GUEST_FSFREEZE_STATUS_FROZEN) {

        ret = qmp_guest_fsfreeze_thaw(&err);

        if (ret < 0 || err) {

            slog("failed to clean up frozen filesystems");

        }

    }

}
