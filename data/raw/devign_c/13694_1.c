void ff_get_guid(AVIOContext *s, ff_asf_guid *g)

{

    assert(sizeof(*g) == 16);

    avio_read(s, *g, sizeof(*g));

}
