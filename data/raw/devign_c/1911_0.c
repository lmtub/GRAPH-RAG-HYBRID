static void win_chr_readfile(CharDriverState *chr)

{

    WinCharState *s = chr->opaque;

    int ret, err;

    uint8_t buf[1024];

    DWORD size;



    ZeroMemory(&s->orecv, sizeof(s->orecv));

    s->orecv.hEvent = s->hrecv;

    ret = ReadFile(s->hcom, buf, s->len, &size, &s->orecv);

    if (!ret) {

        err = GetLastError();

        if (err == ERROR_IO_PENDING) {

            ret = GetOverlappedResult(s->hcom, &s->orecv, &size, TRUE);

        }

    }



    if (size > 0) {

        qemu_chr_read(chr, buf, size);

    }

}
