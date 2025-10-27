ff_vorbis_comment(AVFormatContext * as, AVDictionary **m, const uint8_t *buf, int size)

{

    const uint8_t *p = buf;

    const uint8_t *end = buf + size;

    unsigned n, j;

    int s;



    if (size < 8) /* must have vendor_length and user_comment_list_length */

        return -1;



    s = bytestream_get_le32(&p);



    if (end - p - 4 < s || s < 0)

        return -1;



    p += s;



    n = bytestream_get_le32(&p);



    while (end - p >= 4 && n > 0) {

        const char *t, *v;

        int tl, vl;



        s = bytestream_get_le32(&p);



        if (end - p < s || s < 0)

            break;



        t = p;

        p += s;

        n--;



        v = memchr(t, '=', s);

        if (!v)

            continue;



        tl = v - t;

        vl = s - tl - 1;

        v++;



        if (tl && vl) {

            char *tt, *ct;



            tt = av_malloc(tl + 1);

            ct = av_malloc(vl + 1);

            if (!tt || !ct) {

                av_freep(&tt);


                av_log(as, AV_LOG_WARNING, "out-of-memory error. skipping VorbisComment tag.\n");

                continue;

            }



            for (j = 0; j < tl; j++)

                tt[j] = av_toupper(t[j]);

            tt[tl] = 0;



            memcpy(ct, v, vl);

            ct[vl] = 0;



            if (!strcmp(tt, "METADATA_BLOCK_PICTURE")) {

                int ret;

                char *pict = av_malloc(vl);



                if (!pict) {

                    av_log(as, AV_LOG_WARNING, "out-of-memory error. Skipping cover art block.\n");


                    continue;

                }

                if ((ret = av_base64_decode(pict, ct, vl)) > 0)

                    ret = ff_flac_parse_picture(as, pict, ret);

                av_freep(&pict);


                if (ret < 0) {

                    av_log(as, AV_LOG_WARNING, "Failed to parse cover art block.\n");

                    continue;

                }

            } else if (!ogm_chapter(as, tt, ct))

                av_dict_set(m, tt, ct,

                                   AV_DICT_DONT_STRDUP_KEY |

                                   AV_DICT_DONT_STRDUP_VAL);

        }

    }



    if (p != end)

        av_log(as, AV_LOG_INFO, "%ti bytes of comment header remain\n", end-p);

    if (n > 0)

        av_log(as, AV_LOG_INFO,

               "truncated comment header, %i comments not found\n", n);



    ff_metadata_conv(m, NULL, ff_vorbiscomment_metadata_conv);



    return 0;

}