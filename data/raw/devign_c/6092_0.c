static CharDriverState *qemu_chr_open_spice_port(const char *id,

                                                 ChardevBackend *backend,

                                                 ChardevReturn *ret,

                                                 Error **errp)

{

    const char *name = backend->spiceport->fqdn;

    CharDriverState *chr;

    SpiceCharDriver *s;



    if (name == NULL) {

        fprintf(stderr, "spice-qemu-char: missing name parameter\n");

        return NULL;

    }



    chr = chr_open("port", spice_port_set_fe_open);

    s = chr->opaque;

    s->sin.portname = g_strdup(name);



    return chr;

}
