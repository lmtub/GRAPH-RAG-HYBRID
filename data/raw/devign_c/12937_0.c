static int dv_read_header(AVFormatContext *s,

                          AVFormatParameters *ap)

{

    RawDVContext *c = s->priv_data;

    c->dv_demux = dv_init_demux(s);

   

    return c->dv_demux ? 0 : -1;

}
