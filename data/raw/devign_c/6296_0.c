static void net_socket_receive_dgram(void *opaque, const uint8_t *buf, size_t size)

{

    NetSocketState *s = opaque;

    sendto(s->fd, buf, size, 0,

           (struct sockaddr *)&s->dgram_dst, sizeof(s->dgram_dst));

}
