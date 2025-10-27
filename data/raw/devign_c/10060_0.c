int ffurl_register_protocol(URLProtocol *protocol)

{

    URLProtocol **p;

    p = &first_protocol;

    while (*p != NULL)

        p = &(*p)->next;

    *p             = protocol;

    protocol->next = NULL;

    return 0;

}
