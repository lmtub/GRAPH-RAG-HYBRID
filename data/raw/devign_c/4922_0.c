static void check_suspend_mode(GuestSuspendMode mode, Error **errp)

{

    SYSTEM_POWER_CAPABILITIES sys_pwr_caps;

    Error *local_err = NULL;



    ZeroMemory(&sys_pwr_caps, sizeof(sys_pwr_caps));

    if (!GetPwrCapabilities(&sys_pwr_caps)) {

        error_setg(&local_err, QERR_QGA_COMMAND_FAILED,

                   "failed to determine guest suspend capabilities");

        goto out;

    }



    switch (mode) {

    case GUEST_SUSPEND_MODE_DISK:

        if (!sys_pwr_caps.SystemS4) {

            error_setg(&local_err, QERR_QGA_COMMAND_FAILED,

                       "suspend-to-disk not supported by OS");

        }

        break;

    case GUEST_SUSPEND_MODE_RAM:

        if (!sys_pwr_caps.SystemS3) {

            error_setg(&local_err, QERR_QGA_COMMAND_FAILED,

                       "suspend-to-ram not supported by OS");

        }

        break;

    default:

        error_setg(&local_err, QERR_INVALID_PARAMETER_VALUE, "mode",

                   "GuestSuspendMode");

    }



out:

    if (local_err) {

        error_propagate(errp, local_err);

    }

}
