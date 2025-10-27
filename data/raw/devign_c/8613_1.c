static void baum_chr_open(Chardev *chr,

                          ChardevBackend *backend,

                          bool *be_opened,

                          Error **errp)

{

    BaumChardev *baum = BAUM_CHARDEV(chr);

    brlapi_handle_t *handle;



    handle = g_malloc0(brlapi_getHandleSize());

    baum->brlapi = handle;



    baum->brlapi_fd = brlapi__openConnection(handle, NULL, NULL);

    if (baum->brlapi_fd == -1) {

        error_setg(errp, "brlapi__openConnection: %s",

                   brlapi_strerror(brlapi_error_location()));

        g_free(handle);


        return;

    }

    baum->deferred_init = 0;



    baum->cellCount_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, baum_cellCount_timer_cb, baum);



    qemu_set_fd_handler(baum->brlapi_fd, baum_chr_read, NULL, baum);

}