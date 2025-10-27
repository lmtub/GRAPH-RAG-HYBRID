static void raw_close_fd_pool(BDRVRawState *s)

{

    int i;



    for (i = 0; i < RAW_FD_POOL_SIZE; i++) {

        if (s->fd_pool[i] != -1) {

            close(s->fd_pool[i]);

            s->fd_pool[i] = -1;

        }

    }

}
