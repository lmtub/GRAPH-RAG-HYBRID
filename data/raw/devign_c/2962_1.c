static int rtp_parse_mp4_au(RTPDemuxContext *s, const uint8_t *buf)

{

    int au_headers_length, au_header_size, i;

    GetBitContext getbitcontext;

    RTPPayloadData *infos;



    infos = s->rtp_payload_data;



    if (infos == NULL)

        return -1;



    /* decode the first 2 bytes where the AUHeader sections are stored

       length in bits */

    au_headers_length = AV_RB16(buf);



    if (au_headers_length > RTP_MAX_PACKET_LENGTH)

      return -1;



    infos->au_headers_length_bytes = (au_headers_length + 7) / 8;



    /* skip AU headers length section (2 bytes) */

    buf += 2;



    init_get_bits(&getbitcontext, buf, infos->au_headers_length_bytes * 8);



    /* XXX: Wrong if optionnal additional sections are present (cts, dts etc...) */

    au_header_size = infos->sizelength + infos->indexlength;

    if (au_header_size <= 0 || (au_headers_length % au_header_size != 0))

        return -1;



    infos->nb_au_headers = au_headers_length / au_header_size;



    infos->au_headers = av_malloc(sizeof(struct AUHeaders) * infos->nb_au_headers);





    /* XXX: We handle multiple AU Section as only one (need to fix this for interleaving)

       In my test, the FAAD decoder does not behave correctly when sending each AU one by one

       but does when sending the whole as one big packet...  */

    infos->au_headers[0].size = 0;

    infos->au_headers[0].index = 0;

    for (i = 0; i < infos->nb_au_headers; ++i) {

        infos->au_headers[0].size += get_bits_long(&getbitcontext, infos->sizelength);

        infos->au_headers[0].index = get_bits_long(&getbitcontext, infos->indexlength);




    infos->nb_au_headers = 1;



    return 0;
