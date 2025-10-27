static GIOStatus ga_channel_write(GAChannel *c, const char *buf, size_t size,

                                  size_t *count)

{

    GIOStatus status;

    OVERLAPPED ov = {0};

    BOOL ret;

    DWORD written;



    ov.hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

    ret = WriteFile(c->handle, buf, size, &written, &ov);

    if (!ret) {

        if (GetLastError() == ERROR_IO_PENDING) {

            /* write is pending */

            ret = GetOverlappedResult(c->handle, &ov, &written, TRUE);

            if (!ret) {

                if (!GetLastError()) {

                    status = G_IO_STATUS_AGAIN;

                } else {

                    status = G_IO_STATUS_ERROR;


            } else {

                /* write is complete */

                status = G_IO_STATUS_NORMAL;

                *count = written;


        } else {

            status = G_IO_STATUS_ERROR;


    } else {

        /* write returned immediately */

        status = G_IO_STATUS_NORMAL;

        *count = written;








    return status;
