static void acquire_privilege(const char *name, Error **errp)

{

    HANDLE token = NULL;

    TOKEN_PRIVILEGES priv;

    Error *local_err = NULL;



    if (OpenProcessToken(GetCurrentProcess(),

        TOKEN_ADJUST_PRIVILEGES|TOKEN_QUERY, &token))

    {

        if (!LookupPrivilegeValue(NULL, name, &priv.Privileges[0].Luid)) {

            error_setg(&local_err, QERR_QGA_COMMAND_FAILED,

                       "no luid for requested privilege");

            goto out;

        }



        priv.PrivilegeCount = 1;

        priv.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;



        if (!AdjustTokenPrivileges(token, FALSE, &priv, 0, NULL, 0)) {

            error_setg(&local_err, QERR_QGA_COMMAND_FAILED,

                       "unable to acquire requested privilege");

            goto out;

        }



    } else {

        error_setg(&local_err, QERR_QGA_COMMAND_FAILED,

                   "failed to open privilege token");

    }



out:

    if (token) {

        CloseHandle(token);

    }

    if (local_err) {

        error_propagate(errp, local_err);

    }

}
