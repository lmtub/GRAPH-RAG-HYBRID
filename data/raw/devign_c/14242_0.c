static CharDriverState *chr_baum_init(const char *id,

                                      ChardevBackend *backend,

                                      ChardevReturn *ret,

                                      Error **errp)

{

    BaumDriverState *baum;

    CharDriverState *chr;

    brlapi_handle_t *handle;

#if defined(CONFIG_SDL)

#if SDL_COMPILEDVERSION < SDL_VERSIONNUM(2, 0, 0)

    SDL_SysWMinfo info;

#endif

#endif

    int tty;



    baum = g_malloc0(sizeof(BaumDriverState));

    baum->chr = chr = qemu_chr_alloc();



    chr->opaque = baum;

    chr->chr_write = baum_write;

    chr->chr_accept_input = baum_accept_input;

    chr->chr_close = baum_close;



    handle = g_malloc0(brlapi_getHandleSize());

    baum->brlapi = handle;



    baum->brlapi_fd = brlapi__openConnection(handle, NULL, NULL);

    if (baum->brlapi_fd == -1) {

        error_setg(errp, "brlapi__openConnection: %s",

                   brlapi_strerror(brlapi_error_location()));

        goto fail_handle;

    }



    baum->cellCount_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, baum_cellCount_timer_cb, baum);



    if (brlapi__getDisplaySize(handle, &baum->x, &baum->y) == -1) {

        error_setg(errp, "brlapi__getDisplaySize: %s",

                   brlapi_strerror(brlapi_error_location()));

        goto fail;

    }



#if defined(CONFIG_SDL)

#if SDL_COMPILEDVERSION < SDL_VERSIONNUM(2, 0, 0)

    memset(&info, 0, sizeof(info));

    SDL_VERSION(&info.version);

    if (SDL_GetWMInfo(&info))

        tty = info.info.x11.wmwindow;

    else

#endif

#endif

        tty = BRLAPI_TTY_DEFAULT;



    if (brlapi__enterTtyMode(handle, tty, NULL) == -1) {

        error_setg(errp, "brlapi__enterTtyMode: %s",

                   brlapi_strerror(brlapi_error_location()));

        goto fail;

    }



    qemu_set_fd_handler(baum->brlapi_fd, baum_chr_read, NULL, baum);



    return chr;



fail:

    timer_free(baum->cellCount_timer);

    brlapi__closeConnection(handle);

fail_handle:

    g_free(handle);

    g_free(chr);

    g_free(baum);

    return NULL;

}
