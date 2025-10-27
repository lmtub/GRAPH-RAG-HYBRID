static HTTPContext *find_rtp_session_with_url(const char *url, 

                                              const char *session_id)

{

    HTTPContext *rtp_c;

    char path1[1024];

    const char *path;

    char buf[1024];

    int s;



    rtp_c = find_rtp_session(session_id);

    if (!rtp_c)

        return NULL;



    /* find which url is asked */

    url_split(NULL, 0, NULL, 0, NULL, path1, sizeof(path1), url);

    path = path1;

    if (*path == '/')

        path++;

    if(!strcmp(path, rtp_c->stream->filename)) return rtp_c;

    for(s=0; s<rtp_c->stream->nb_streams; ++s) {

      snprintf(buf, sizeof(buf), "%s/streamid=%d",

        rtp_c->stream->filename, s);

      if(!strncmp(path, buf, sizeof(buf))) {

    // XXX: Should we reply with RTSP_STATUS_ONLY_AGGREGATE if nb_streams>1?

        return rtp_c;

      }

    }

    return NULL;

}
