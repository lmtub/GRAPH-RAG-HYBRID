static int local_mksock(FsContext *ctx2, const char *path)

{

    struct sockaddr_un addr;

    int s;



    addr.sun_family = AF_UNIX;

    snprintf(addr.sun_path, 108, "%s", rpath(ctx2, path));



    s = socket(PF_UNIX, SOCK_STREAM, 0);

    if (s == -1) {

        return -1;

    }



    if (bind(s, (struct sockaddr *)&addr, sizeof(addr))) {

        close(s);

        return -1;

    }



    close(s);

    return 0;

}
