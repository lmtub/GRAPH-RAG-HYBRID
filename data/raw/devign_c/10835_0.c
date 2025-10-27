const char *avio_enum_protocols(void **opaque, int output)

{

    URLProtocol *p;

    *opaque = ffurl_protocol_next(*opaque);

    if (!(p = *opaque))

        return NULL;

    if ((output && p->url_write) || (!output && p->url_read))

        return p->name;

    return avio_enum_protocols(opaque, output);

}
