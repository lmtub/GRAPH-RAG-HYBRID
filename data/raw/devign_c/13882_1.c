int chr_baum_init(QemuOpts *opts, CharDriverState **_chr)

{

    BaumDriverState *baum;

    CharDriverState *chr;

    brlapi_handle_t *handle;

#ifdef CONFIG_SDL

    SDL_SysWMinfo info;

#endif

    int tty;



    baum = g_malloc0(sizeof(BaumDriverState));

    baum->chr = chr = g_malloc0(sizeof(CharDriverState));



    chr->opaque = baum;

    chr->chr_write = baum_write;

    chr->chr_accept_input = baum_accept_input;

    chr->chr_close = baum_close;



    handle = g_malloc0(brlapi_getHandleSize());

    baum->brlapi = handle;



    baum->brlapi_fd = brlapi__openConnection(handle, NULL, NULL);

    if (baum->brlapi_fd == -1) {

        brlapi_perror("baum_init: brlapi_openConnection");

        goto fail_handle;

    }



    baum->cellCount_timer = qemu_new_timer_ns(vm_clock, baum_cellCount_timer_cb, baum);



    if (brlapi__getDisplaySize(handle, &baum->x, &baum->y) == -1) {

        brlapi_perror("baum_init: brlapi_getDisplaySize");

        goto fail;

    }



#ifdef CONFIG_SDL

    memset(&info, 0, sizeof(info));

    SDL_VERSION(&info.version);

    if (SDL_GetWMInfo(&info))

        tty = info.info.x11.wmwindow;

    else

#endif

        tty = BRLAPI_TTY_DEFAULT;



    if (brlapi__enterTtyMode(handle, tty, NULL) == -1) {

        brlapi_perror("baum_init: brlapi_enterTtyMode");

        goto fail;

    }



    qemu_set_fd_handler(baum->brlapi_fd, baum_chr_read, NULL, baum);



    qemu_chr_generic_open(chr);



    *_chr = chr;

    return 0;



fail:

    qemu_free_timer(baum->cellCount_timer);

    brlapi__closeConnection(handle);

fail_handle:

    g_free(handle);

    g_free(chr);

    g_free(baum);

    return -EIO;

}
