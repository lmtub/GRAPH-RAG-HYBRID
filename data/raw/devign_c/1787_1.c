static int check_for_evdev(void)

{

    SDL_SysWMinfo info;

    XkbDescPtr desc;

    int has_evdev = 0;

    const char *keycodes;



    SDL_VERSION(&info.version);

    if (!SDL_GetWMInfo(&info))

        return 0;



    desc = XkbGetKeyboard(info.info.x11.display,

                          XkbGBN_AllComponentsMask,

                          XkbUseCoreKbd);

    if (desc == NULL || desc->names == NULL)

        return 0;



    keycodes = XGetAtomName(info.info.x11.display, desc->names->keycodes);

    if (keycodes == NULL)

        fprintf(stderr, "could not lookup keycode name\n");

    else if (strstart(keycodes, "evdev", NULL))

        has_evdev = 1;

    else if (!strstart(keycodes, "xfree86", NULL))

        fprintf(stderr,

                "unknown keycodes `%s', please report to qemu-devel@nongnu.org\n",

                keycodes);



    XkbFreeClientMap(desc, XkbGBN_AllComponentsMask, True);



    return has_evdev;

}
