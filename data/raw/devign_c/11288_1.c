static int parse_vdiname(BDRVSheepdogState *s, const char *filename,

                         char *vdi, uint32_t *snapid, char *tag)

{

    char *p, *q, *uri;

    const char *host_spec, *vdi_spec;

    int nr_sep, ret;



    strstart(filename, "sheepdog:", &filename);

    p = q = g_strdup(filename);



    /* count the number of separators */

    nr_sep = 0;

    while (*p) {

        if (*p == ':') {

            nr_sep++;

        }

        p++;

    }

    p = q;



    /* use the first two tokens as host_spec. */

    if (nr_sep >= 2) {

        host_spec = p;

        p = strchr(p, ':');

        p++;

        p = strchr(p, ':');

        *p++ = '\0';

    } else {

        host_spec = "";

    }



    vdi_spec = p;



    p = strchr(vdi_spec, ':');

    if (p) {

        *p++ = '#';

    }



    uri = g_strdup_printf("sheepdog://%s/%s", host_spec, vdi_spec);



    ret = sd_parse_uri(s, uri, vdi, snapid, tag);



    g_free(q);

    g_free(uri);



    return ret;

}
