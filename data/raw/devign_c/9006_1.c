static DWORD WINAPI do_suspend(LPVOID opaque)

{

    GuestSuspendMode *mode = opaque;

    DWORD ret = 0;



    if (!SetSuspendState(*mode == GUEST_SUSPEND_MODE_DISK, TRUE, TRUE)) {

        slog("failed to suspend guest, %s", GetLastError());

        ret = -1;

    }

    g_free(mode);

    return ret;

}
