void ffserver_parse_acl_row(FFServerStream *stream, FFServerStream* feed,
                            FFServerIPAddressACL *ext_acl,
                            const char *p, const char *filename, int line_num)
{
    char arg[1024];
    FFServerIPAddressACL acl;
    FFServerIPAddressACL *nacl;
    FFServerIPAddressACL **naclp;
    ffserver_get_arg(arg, sizeof(arg), &p);
    if (av_strcasecmp(arg, "allow") == 0)
        acl.action = IP_ALLOW;
    else if (av_strcasecmp(arg, "deny") == 0)
        acl.action = IP_DENY;
    else {
        fprintf(stderr, "%s:%d: ACL action '%s' should be ALLOW or DENY.\n",
                filename, line_num, arg);
    ffserver_get_arg(arg, sizeof(arg), &p);
    if (resolve_host(&acl.first, arg)) {
        fprintf(stderr,
                "%s:%d: ACL refers to invalid host or IP address '%s'\n",
                filename, line_num, arg);
    acl.last = acl.first;
    ffserver_get_arg(arg, sizeof(arg), &p);
    if (arg[0]) {
        if (resolve_host(&acl.last, arg)) {
            fprintf(stderr,
                    "%s:%d: ACL refers to invalid host or IP address '%s'\n",
                    filename, line_num, arg);
    nacl = av_mallocz(sizeof(*nacl));
    naclp = 0;
    acl.next = 0;
    *nacl = acl;
    if (stream)
        naclp = &stream->acl;
    else if (feed)
        naclp = &feed->acl;
    else if (ext_acl)
        naclp = &ext_acl;
    else
        fprintf(stderr, "%s:%d: ACL found not in <Stream> or <Feed>\n",
                filename, line_num);
    if (naclp) {
        while (*naclp)
            naclp = &(*naclp)->next;
        *naclp = nacl;
    } else
        av_free(nacl);
bail:
  return;