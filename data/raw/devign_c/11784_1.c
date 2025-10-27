static inline int parse_command_line(AVFormatContext *s, const char *line,
                                     int linelen, char *uri, int urisize,
                                     char *method, int methodsize,
                                     enum RTSPMethod *methodcode)
{
    RTSPState *rt = s->priv_data;
    const char *linept, *searchlinept;
    linept = strchr(line, ' ');
    if (linept - line > methodsize - 1) {
        av_log(s, AV_LOG_ERROR, "Method string too long\n");
        return AVERROR(EIO);
    }
    memcpy(method, line, linept - line);
    method[linept - line] = '\0';
    linept++;
    if (!strcmp(method, "ANNOUNCE"))
        *methodcode = ANNOUNCE;
    else if (!strcmp(method, "OPTIONS"))
        *methodcode = OPTIONS;
    else if (!strcmp(method, "RECORD"))
        *methodcode = RECORD;
    else if (!strcmp(method, "SETUP"))
        *methodcode = SETUP;
    else if (!strcmp(method, "PAUSE"))
        *methodcode = PAUSE;
    else if (!strcmp(method, "TEARDOWN"))
        *methodcode = TEARDOWN;
    else
        *methodcode = UNKNOWN;
    /* Check method with the state  */
    if (rt->state == RTSP_STATE_IDLE) {
        if ((*methodcode != ANNOUNCE) && (*methodcode != OPTIONS)) {
            av_log(s, AV_LOG_ERROR, "Unexpected command in Idle State %s\n",
                   line);
            return AVERROR_PROTOCOL_NOT_FOUND;
        }
    } else if (rt->state == RTSP_STATE_PAUSED) {
        if ((*methodcode != OPTIONS) && (*methodcode != RECORD)
            && (*methodcode != SETUP)) {
            av_log(s, AV_LOG_ERROR, "Unexpected command in Paused State %s\n",
                   line);
            return AVERROR_PROTOCOL_NOT_FOUND;
        }
    } else if (rt->state == RTSP_STATE_STREAMING) {
        if ((*methodcode != PAUSE) && (*methodcode != OPTIONS)
            && (*methodcode != TEARDOWN)) {
            av_log(s, AV_LOG_ERROR, "Unexpected command in Streaming State"
                   " %s\n", line);
            return AVERROR_PROTOCOL_NOT_FOUND;
        }
    } else {
        av_log(s, AV_LOG_ERROR, "Unexpected State [%d]\n", rt->state);
        return AVERROR_BUG;
    }
    searchlinept = strchr(linept, ' ');
    if (!searchlinept) {
        av_log(s, AV_LOG_ERROR, "Error parsing message URI\n");
    }
    if (searchlinept - linept > urisize - 1) {
        av_log(s, AV_LOG_ERROR, "uri string length exceeded buffer size\n");
        return AVERROR(EIO);
    }
    memcpy(uri, linept, searchlinept - linept);
    uri[searchlinept - linept] = '\0';
    if (strcmp(rt->control_uri, uri)) {
        char host[128], path[512], auth[128];
        int port;
        char ctl_host[128], ctl_path[512], ctl_auth[128];
        int ctl_port;
        av_url_split(NULL, 0, auth, sizeof(auth), host, sizeof(host), &port,
                     path, sizeof(path), uri);
        av_url_split(NULL, 0, ctl_auth, sizeof(ctl_auth), ctl_host,
                     sizeof(ctl_host), &ctl_port, ctl_path, sizeof(ctl_path),
                     rt->control_uri);
        if (strcmp(host, ctl_host))
            av_log(s, AV_LOG_INFO, "Host %s differs from expected %s\n",
                   host, ctl_host);
        if (strcmp(path, ctl_path) && *methodcode != SETUP)
            av_log(s, AV_LOG_WARNING, "WARNING: Path %s differs from expected"
                   " %s\n", path, ctl_path);
        if (*methodcode == ANNOUNCE) {
            av_log(s, AV_LOG_INFO,
                   "Updating control URI to %s\n", uri);
            av_strlcpy(rt->control_uri, uri, sizeof(rt->control_uri));
        }
    }
    linept = searchlinept + 1;
    if (!av_strstart(linept, "RTSP/1.0", NULL)) {
        av_log(s, AV_LOG_ERROR, "Error parsing protocol or version\n");
        return AVERROR_PROTOCOL_NOT_FOUND;
    }
    return 0;
}