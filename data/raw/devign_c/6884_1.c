static void net_socket_send(void *opaque)

{

    NetSocketState *s = opaque;

    int l, size, err;

    uint8_t buf1[4096];

    const uint8_t *buf;



    size = recv(s->fd, buf1, sizeof(buf1), 0);

    if (size < 0) {

        err = socket_error();

        if (err != EWOULDBLOCK)

            goto eoc;

    } else if (size == 0) {

        /* end of connection */

    eoc:

        qemu_set_fd_handler(s->fd, NULL, NULL, NULL);

        closesocket(s->fd);

        return;

    }

    buf = buf1;

    while (size > 0) {

        /* reassemble a packet from the network */

        switch(s->state) {

        case 0:

            l = 4 - s->index;

            if (l > size)

                l = size;

            memcpy(s->buf + s->index, buf, l);

            buf += l;

            size -= l;

            s->index += l;

            if (s->index == 4) {

                /* got length */

                s->packet_len = ntohl(*(uint32_t *)s->buf);

                s->index = 0;

                s->state = 1;

            }

            break;

        case 1:

            l = s->packet_len - s->index;

            if (l > size)

                l = size;

            memcpy(s->buf + s->index, buf, l);

            s->index += l;

            buf += l;

            size -= l;

            if (s->index >= s->packet_len) {

                qemu_send_packet(s->vc, s->buf, s->packet_len);

                s->index = 0;

                s->state = 0;

            }

            break;

        }

    }

}
