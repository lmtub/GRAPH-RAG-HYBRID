static int win_chr_pipe_init(CharDriverState *chr, const char *filename,

                             Error **errp)

{

    WinCharState *s = chr->opaque;

    OVERLAPPED ov;

    int ret;

    DWORD size;

    char openname[CHR_MAX_FILENAME_SIZE];



    s->fpipe = TRUE;



    s->hsend = CreateEvent(NULL, TRUE, FALSE, NULL);

    if (!s->hsend) {

        error_setg(errp, "Failed CreateEvent");

        goto fail;

    }

    s->hrecv = CreateEvent(NULL, TRUE, FALSE, NULL);

    if (!s->hrecv) {

        error_setg(errp, "Failed CreateEvent");

        goto fail;

    }



    snprintf(openname, sizeof(openname), "\\\\.\\pipe\\%s", filename);

    s->hcom = CreateNamedPipe(openname, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,

                              PIPE_TYPE_BYTE | PIPE_READMODE_BYTE |

                              PIPE_WAIT,

                              MAXCONNECT, NSENDBUF, NRECVBUF, NTIMEOUT, NULL);

    if (s->hcom == INVALID_HANDLE_VALUE) {

        error_setg(errp, "Failed CreateNamedPipe (%lu)", GetLastError());

        s->hcom = NULL;

        goto fail;

    }



    ZeroMemory(&ov, sizeof(ov));

    ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    ret = ConnectNamedPipe(s->hcom, &ov);

    if (ret) {

        error_setg(errp, "Failed ConnectNamedPipe");

        goto fail;

    }



    ret = GetOverlappedResult(s->hcom, &ov, &size, TRUE);

    if (!ret) {

        error_setg(errp, "Failed GetOverlappedResult");

        if (ov.hEvent) {

            CloseHandle(ov.hEvent);

            ov.hEvent = NULL;

        }

        goto fail;

    }



    if (ov.hEvent) {

        CloseHandle(ov.hEvent);

        ov.hEvent = NULL;

    }

    qemu_add_polling_cb(win_chr_pipe_poll, chr);

    return 0;



 fail:

    win_chr_close(chr);

    return -1;

}
