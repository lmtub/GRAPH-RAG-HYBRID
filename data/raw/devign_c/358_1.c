void qmp_guest_shutdown(bool has_mode, const char *mode, Error **err)

{

    UINT shutdown_flag = EWX_FORCE;



    slog("guest-shutdown called, mode: %s", mode);



    if (!has_mode || strcmp(mode, "powerdown") == 0) {

        shutdown_flag |= EWX_POWEROFF;

    } else if (strcmp(mode, "halt") == 0) {

        shutdown_flag |= EWX_SHUTDOWN;

    } else if (strcmp(mode, "reboot") == 0) {

        shutdown_flag |= EWX_REBOOT;

    } else {

        error_set(err, QERR_INVALID_PARAMETER_VALUE, "mode",

                  "halt|powerdown|reboot");

        return;

    }



    /* Request a shutdown privilege, but try to shut down the system

       anyway. */

    acquire_privilege(SE_SHUTDOWN_NAME, err);

    if (error_is_set(err)) {

        return;

    }



    if (!ExitWindowsEx(shutdown_flag, SHTDN_REASON_FLAG_PLANNED)) {

        slog("guest-shutdown failed: %d", GetLastError());

        error_set(err, QERR_UNDEFINED_ERROR);

    }

}
