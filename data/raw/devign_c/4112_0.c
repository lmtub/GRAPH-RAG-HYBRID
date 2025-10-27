static int usb_net_handle_dataout(USBNetState *s, USBPacket *p)

{

    int ret = p->len;

    int sz = sizeof(s->out_buf) - s->out_ptr;

    struct rndis_packet_msg_type *msg =

            (struct rndis_packet_msg_type *) s->out_buf;

    uint32_t len;



#ifdef TRAFFIC_DEBUG

    fprintf(stderr, "usbnet: data out len %u\n", p->len);

    {

        int i;

        fprintf(stderr, ":");

        for (i = 0; i < p->len; i++) {

            if (!(i & 15))

                fprintf(stderr, "\n%04x:", i);

            fprintf(stderr, " %02x", p->data[i]);

        }

        fprintf(stderr, "\n\n");

    }

#endif



    if (sz > ret)

        sz = ret;

    memcpy(&s->out_buf[s->out_ptr], p->data, sz);

    s->out_ptr += sz;



    if (!s->rndis) {

        if (ret < 64) {

            qemu_send_packet(&s->nic->nc, s->out_buf, s->out_ptr);

            s->out_ptr = 0;

        }

        return ret;

    }

    len = le32_to_cpu(msg->MessageLength);

    if (s->out_ptr < 8 || s->out_ptr < len)

        return ret;

    if (le32_to_cpu(msg->MessageType) == RNDIS_PACKET_MSG) {

        uint32_t offs = 8 + le32_to_cpu(msg->DataOffset);

        uint32_t size = le32_to_cpu(msg->DataLength);

        if (offs + size <= len)

            qemu_send_packet(&s->nic->nc, s->out_buf + offs, size);

    }

    s->out_ptr -= len;

    memmove(s->out_buf, &s->out_buf[len], s->out_ptr);



    return ret;

}
