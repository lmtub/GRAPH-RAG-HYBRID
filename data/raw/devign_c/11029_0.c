URLProtocol *ffurl_protocol_next(const URLProtocol *prev)

{

    return prev ? prev->next : first_protocol;

}
