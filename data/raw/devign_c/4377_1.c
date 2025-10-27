int qemu_chr_open_spice(QemuOpts *opts, CharDriverState **_chr)

{

    CharDriverState *chr;

    SpiceCharDriver *s;

    const char* name = qemu_opt_get(opts, "name");

    uint32_t debug = qemu_opt_get_number(opts, "debug", 0);

    const char** psubtype = spice_server_char_device_recognized_subtypes();

    const char *subtype = NULL;



    if (name == NULL) {

        fprintf(stderr, "spice-qemu-char: missing name parameter\n");

        print_allowed_subtypes();

        return -EINVAL;

    }

    for(;*psubtype != NULL; ++psubtype) {

        if (strcmp(name, *psubtype) == 0) {

            subtype = *psubtype;

            break;

        }

    }

    if (subtype == NULL) {

        fprintf(stderr, "spice-qemu-char: unsupported name\n");

        print_allowed_subtypes();

        return -EINVAL;

    }



    chr = g_malloc0(sizeof(CharDriverState));

    s = g_malloc0(sizeof(SpiceCharDriver));

    s->chr = chr;

    s->debug = debug;

    s->active = false;

    s->sin.subtype = subtype;

    chr->opaque = s;

    chr->chr_write = spice_chr_write;

    chr->chr_close = spice_chr_close;

    chr->chr_guest_open = spice_chr_guest_open;

    chr->chr_guest_close = spice_chr_guest_close;



#if SPICE_SERVER_VERSION < 0x000901

    /* See comment in vmc_state() */

    if (strcmp(subtype, "vdagent") == 0) {

        qemu_chr_generic_open(chr);

    }

#endif



    *_chr = chr;

    return 0;

}
